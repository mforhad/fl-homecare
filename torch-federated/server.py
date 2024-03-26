from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from data.dataloader import fl_config
from config.hc_strategy import HomecareStrategy

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples)
        }


# Define strategy
# strategy = fl.server.strategy.FedAvg(
strategy = HomecareStrategy(
    fraction_fit=fl_config.fraction_fit,
    fraction_evaluate=fl_config.fraction_eval,
    min_fit_clients=fl_config.min_fit_clients,
    min_evaluate_clients=fl_config.min_evaluate_clients,
    min_available_clients=fl_config.num_clients,
    evaluate_metrics_aggregation_fn=weighted_average
)
# strategy = HomecareStrategy(
#     fraction_fit=0.5,
#     fraction_evaluate=0.5,
#     min_fit_clients=1,
#     min_evaluate_clients=1,
#     min_available_clients=2,
#     evaluate_metrics_aggregation_fn=weighted_average
# )


# Define Flower configuration
server_config = fl.server.ServerConfig(
    num_rounds=fl_config.num_rounds,
    round_timeout=fl_config.round_timeout,
)


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=server_config,
    strategy=strategy,
)