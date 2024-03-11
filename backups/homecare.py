import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import numpy as np

import keras
from keras.layers import Input, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar
from typing import Dict, List, Tuple
from mak import homecare_utils as LeNet

parser = argparse.ArgumentParser(description="Tutorial on using multi-node Flower Simulation with PyTorch")

# Add your command line arguments here if needed

NUM_CLIENTS = 100

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument("--multi_node", type=bool, default=False, help="Use in multi-node mode or not")
parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds.")


def lr_schedule(epoch, lr):
    if epoch > 70 and \
            (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.model = LeNet.create_model(input_shape=self.x_train.shape[1:])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config: Dict[str, Scalar]):
        return self.model.get_weights()

    def fit(self, parameters, config: dict):
        """Train the model on the client."""
        self.model.set_weights(parameters)

        # Convert datasets to numpy arrays (if needed)
        # x_train, y_train, x_val, y_val = LeNet.load_data()
        # history = LeNet.train_model(self.model, x_train, y_train, x_val, y_val, epochs=5, batch_size=32)
        lr_scheduler = LearningRateScheduler(lr_schedule) # Dynamic adjustment learning rate
        history = self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=5, validation_data=(self.x_test, self.y_test),
                        callbacks=[lr_scheduler])

        return self.get_parameters(), len(self.x_train), {"history": history.history}

    def evaluate(self, parameters, config: dict):
        """Evaluate the current model on the client."""
        # self.model = LeNet.create_model(input_shape=(28, 2))  # Adjust input shape as needed
        self.model.set_weights(parameters)

        # Convert testset to numpy arrays (if needed)
        # x_train, y_train, x_test, y_test = LeNet.load_data()
        # loss, accuracy = LeNet.test_model(self.model, x_test, y_test)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test) # test the model

        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}


def get_client_fn(train_partitions, val_partitions):
    def client_fn(cid: int) -> fl.client.Client:
        """Client function used by the virtual clients in the simulation."""
        print("################### cid = ", cid)
        # Load and prepare the dataset
        x_train, y_train, x_test, y_test = prepare_dataset()


        # Access the corresponding partition for this client
        # trainset = trainset[int(cid) % len(trainset)]
        # valset = valset[int(cid) % len(valset)]

        # trainset, valset = train_partitions[int(cid) % len(train_partitions)], val_partitions[int(cid) % len(train_partitions)]

        # return trainset_partition, valset_partition
        # return FlowerClient(trainset_partition, valset_partition, train_shape)
        return FlowerClient(x_train, y_train, x_test, y_test).to_client()

    return client_fn


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # Number of local epochs done by clients
        "batch_size": 32,  # Batch size to use by clients during fit()
    }
    return config


def weighted_average(metrics: List[List[Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    # total_examples = sum([num_examples for _, num_examples, _ in metrics])
    # total_loss = sum([loss * num_examples for loss, num_examples, _ in metrics])
    # total_accuracy = sum([acc * num_examples for _, num_examples, acc in metrics])

    # # Aggregate and return custom metric (weighted average)
    # return {"loss": total_loss / total_examples, "accuracy": total_accuracy / total_examples}
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(x_test, y_test):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArray, config: Dict[str, Scalar]
    ):
        """Use the entire test set for evaluation."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = LeNet.create_model(input_shape=(28, 2))  # Adjust input shape as needed
        # model.set_weights(parameters)
        # model.to(device) # removeif

        # x_test, y_test = testset
        loss, accuracy = model.evaluate(x_test, y_test) # test the model
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate


def prepare_dataset():
    """Load and partition the dataset."""
    x_train, y_train, _, x_test, y_test, _ = LeNet.load_data()
    x_train = x_train[: 16000]
    y_train = y_train[: 16000]
    x_test = x_test[: 16000]
    y_test = y_test[: 16000]

    y_train = keras.utils.to_categorical(y_train, num_classes=2) # Convert to two categories
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    train_size = int(len(x_train) * 1)  # 90% for training, 10% for validation
    val_size = len(x_train) - train_size

    # trainset = (x_train[:train_size], y_train[:train_size])
    # valset = (x_train[train_size:], y_train[train_size:])
    trainset = (x_train, y_train)
    valset = (x_test, y_test)
    testset = (x_test, y_test)

    # return trainset, valset, testset
    return x_train, y_train, x_test, y_test


# def prepare_dataset():
#     """Load and partition the dataset."""
#     x_train, y_train, _, x_test, y_test, _ = LeNet.load_data()
#     x_train = x_train[: 16000]
#     y_train = y_train[: 16000]
#     x_test = x_test[: 16000]
#     y_test = y_test[: 16000]
#
#     y_train = keras.utils.to_categorical(y_train, num_classes=2) # Convert to two categories
#     y_test = keras.utils.to_categorical(y_test, num_classes=2)
#
#     num_data_per_client = len(x_train) // NUM_CLIENTS
#     partition_len = [num_data_per_client] * NUM_CLIENTS
#
#     x_trainsets = random_split(
#         x_train, partition_len, torch.Generator.manual_seed(2023)
#     )
#
#     val_ratio = 0.0
#
#     x_train_part = []
#     y_train_part = []
#     x_test_part = []
#     y_test_part = []
#     for xt_sets in x_trainsets:
#         num_total = len(xt_sets)
#         num_val = int(val_ratio * num_total)
#         num_train = num_total - num_val
#
#         for_xtrain, for_ytrain, for_xval, for_yval = random_split(
#             xt_sets, [num_train, num_train, num_val, num_val], torch.Generator().manual_seed(2023)
#         )
#
#         x_train_part.append(for_xtrain)
#         y_train_part.append(for_ytrain)
#         x_test_part.append(for_xval)
#         y_test_part.append(for_yval)
#
#     # train_size = int(len(x_train) * 0.9)  # 90% for training, 10% for validation
#     # val_size = len(x_train) - train_size
#
#     # trainset = (x_train[:train_size], y_train[:train_size])
#     # valset = (x_train[train_size:], y_train[train_size:])
#     # trainset = (x_train, y_train)
#     # valset = (x_test, y_test)
#     # testset = (x_test, y_test)
#
#     # return trainset, valset, testset
#     return x_train_part, y_train_part, x_test_part, y_test_part


def main():
    # Parse input arguments
    args = parser.parse_args()

    # Download CIFAR-10 dataset and partition it
    # train_partitions, val_partitions, testset= prepare_dataset()
    x_train, y_train, x_test, y_test = prepare_dataset()
    # x_train, y_train, _, x_test, y_test, _ = LeNet.load_data()
    train_partitions = (x_train, y_train)
    val_partitions = testset = (x_test, y_test)

    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(x_test, y_test),  # Global evaluation function
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # if args.multi_node:
    #     ray_init_args = {"address" : "auto","runtime_env" : {"py_modules" : [mak]}} #if multi-node cluster is used
    # else:
    #     ray_init_args = {}

    ray_init_args = {}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(train_partitions, val_partitions),
        num_clients=NUM_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )


if __name__ == "__main__":
    main()

