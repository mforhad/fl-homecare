import torch
import torch.nn.functional as F
import warnings
import argparse

import flwr as fl

from collections import OrderedDict
from model.model import LeNet, DEVICE, train, test
from data.dataloader import load_partition, fl_config

warnings.filterwarnings("ignore", category=UserWarning)

# Get partition id
choice_list = [i for i in range(fl_config.num_clients)]
print("### Choise List : ", choice_list)
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=choice_list,
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
partition_id = parser.parse_args().partition_id


net = LeNet().to(DEVICE)
trainloader, valloader, testloader = load_partition(partition_id=partition_id, batch_size=fl_config.batch_size)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id) -> None:
        super().__init__()
        self.client_id = client_id

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.client_id, net, trainloader, valloader, num_epochs=fl_config.num_epochs)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.client_id, net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(partition_id).to_client(),
)