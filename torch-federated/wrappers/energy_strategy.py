import random
from logging import WARNING
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

class EnergyAwareFedAvg(fl.server.strategy.FedAvg):
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
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        ):
        print("Initializing Strategy.....")
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,            
        )
        self.energy_consumed = energy_consumed
        self.training_time = training_time
        self.available_memory = available_memory
        self.available_gpu_memory = available_gpu_memory
        self.history = {}  # Stores historical data for each client

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
            energy_weight = 0.3
            accuracy_weight = 0.4
            time_weight = 0.3

            # Calculate average values (you can customize the calculation)
            avg_energy = sum(data["energy"]) / len(data["energy"])
            avg_accuracy = sum(data["accuracy"]) / len(data["accuracy"])
            avg_time = sum(data["time"]) / len(data["time"])

            # Combine information into a score (customizable logic)
            score = (energy_weight * avg_energy) + (accuracy_weight * avg_accuracy) - (time_weight * avg_time)  # Prioritize lower energy and faster time
            scores[client_id] = score

        # 3. Sort clients based on score (descending for lower energy and faster time)
        sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Select a fraction of clients based on fraction_fit
        selected_clients = [client_id for client_id, _ in sorted_clients[:int(self.fraction_fit * len(available_clients))]]

        return selected_clients

    def aggregate(self, federated_results):
        """Aggregates model updates using FedAvg with additional filtering."""

        # Filter out low-performing clients (optional)
        filtered_results = [
            (client_id, updates) for client_id, updates in federated_results if scores[client_id] > some_threshold
        ]

        # Perform FedAvg aggregation on filtered results
        aggregated_updates = super().aggregate(filtered_results)
        return aggregated_updates