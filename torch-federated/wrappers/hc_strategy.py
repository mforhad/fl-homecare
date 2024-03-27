import random
from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

from data.dataloader import fl_config

class HomecareStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            fraction_fit: float = 0.5,
            fraction_evaluate: float = 0.5,
            min_fit_clients: int = 1,
            min_evaluate_clients: int = 1,
            min_available_clients: int = 2,
            energy_consumed: Optional[float] = 0.0,
            training_time: Optional[float] = 0.0,
            available_memory: Optional[int] = 0,
            available_gpu_memory: Optional[int] = 0,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        ):
        print("Initializing Strategy.....")
        self.energy_consumed = energy_consumed
        self.training_time = training_time
        self.available_memory = available_memory
        self.available_gpu_memory = available_gpu_memory
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn          
        )

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def select_clients(self, ctx, available_clients):
        """Selects clients based on energy consumption, accuracy, and training time."""

        # 1. Collect client information (if not already collected)
        for client in available_clients:
            if client.id not in self.history:
                self.history[client.id] = {"energy": [], "accuracy": [], "time": []}
            energy_consumption = client.get_energy_consumption()  # Replace with actual method
            accuracy = client.get_model_accuracy()  # Replace with actual method
            training_time = client.get_training_time()  # Replace with actual method
            self.history[client.id]["energy"].append(energy_consumption)
            self.history[client.id]["accuracy"].append(accuracy)
            self.history[client.id]["time"].append(training_time)

        # 2. Calculate weighted score for each client
        scores = {}
        for client_id, data in self.history.items():
            # Define weightage for each parameter (optional)
            energy_weight = fl_config.energy_weight
            accuracy_weight = fl_config.accuracy_weight
            time_weight = fl_config.train_time_weight

            # Calculate average values (you can customize the calculation)
            avg_energy = sum(data["energy"]) / len(data["energy"])
            avg_accuracy = sum(data["accuracy"]) / len(data["accuracy"])
            avg_time = sum(data["time"]) / len(data["time"])

            # Combine information into a score (customizable logic)
            score = (accuracy_weight * avg_accuracy) - (energy_weight * avg_energy) - (time_weight * avg_time)  # Prioritize lower energy and faster time
            scores[client_id] = score

        # 3. Sort clients based on score (descending for lower energy and faster time)
        sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Select 10% random clients. Minimum selection = 1

        # 4. 90% of the fraction of clients based on score
        selected_clients = [client_id for client_id, _ in sorted_clients[:int(self.fraction_fit * len(available_clients))]]

        return selected_clients
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated