import random
from typing import List, Dict, Tuple
from flwr.server.strategy import Strategy
from data.dataloader import fl_config

class HomecareStrategy(Strategy):
    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            energy_consumed: float = 0,
            training_time: float = 0,
            available_memory: int = 0,
            available_gpu_memory: int = 0
        ):
        self.fraction_fit = fraction_fit,
        self.fraction_evaluate = fraction_evaluate,
        self.min_fit_clients = min_fit_clients,
        self.min_evaluate_clients = min_evaluate_clients,
        self.min_available_clients = min_available_clients,
        self.energy_consumed = energy_consumed,
        self.training_time = training_time
        self.available_memory = available_memory
        self.available_gpu_memory = available_gpu_memory

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def get_client_fit_weights(self, client_info):
        energy_weight = self.energy_weights.get(client_info["id"], 1.0)
        time_weight = self.time_weights.get(client_info["id"], 1.0)
        hardware_weight = self.hardware_weights.get(client_info["id"], 1.0)

        total_weight = energy_weight + time_weight + hardware_weight

        return {
            client_info["id"]: total_weight,
        }

    def fit_weights(self, results) -> Dict[str, float]:
        weights = {}

        for result in results:
            client_weights = self.get_client_fit_weights(result.client_info)

            for client_id, weight in client_weights.items():
                if client_id not in weights:
                    weights[client_id] = 0.0
                weights[client_id] += weight

        return weights
