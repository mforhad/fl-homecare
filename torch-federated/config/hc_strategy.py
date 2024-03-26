import random
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
        )

