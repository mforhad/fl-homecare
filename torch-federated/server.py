from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from utilities.dataloader import fl_config


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=fl_config.num_clients,
    min_evaluate_clients=fl_config.num_clients,
    min_available_clients=fl_config.num_clients,
    evaluate_metrics_aggregation_fn=weighted_average
    )

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=fl_config.num_rounds),
    strategy=strategy,
)