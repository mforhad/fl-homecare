import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import warnings
import argparse

import flwr as fl

from dataloader import load_partition

from collections import OrderedDict

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.reshape(x, (-1, 900, 2))
        # print(x.shape)
        x = nn.functional.relu(self.conv1(x.permute(0, 2, 1)))
        x = nn.functional.max_pool1d(x, 3)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool1d(x, 3)
        x = x.view(x.size(0), -1)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.softmax(x, dim=1)


def lr_schedule(epoch, lr):
    if epoch > 70 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr


def train(model, train_loader, test_loader, num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch, 0.155))

    model.to(DEVICE)
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # Free up GPU memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)

                loss = criterion(outputs, labels.long())
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Free up GPU memory
                torch.cuda.empty_cache()

            epoch_loss = running_loss / len(test_loader)
            epoch_acc = correct / total

            print(f"Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")

        scheduler.step()


def test(model, test_loader, y_test=None, groups_test=None):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

            loss = criterion(outputs, labels.long())
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Free up GPU memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()

        accuracy = correct / total
        loss = running_loss / len(test_loader)
        print(f"Test Accuracy: {accuracy:.4f}")


    # Save prediction scores
    if y_test is not None:
        print("###### Saving prediction scores ########")
        y_score = []
        model.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                y_score.extend(outputs.cpu().numpy()[:, 1])

        output = pd.DataFrame({"y_true": y_test, "y_score": y_score, "subject": groups_test})
        output.to_csv(os.path.join("output", "homecare_pytorch.csv"), index=False)

    return loss, accuracy


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
partition_id = parser.parse_args().partition_id

# Load model and data (simple CNN, CIFAR-10)
net = LeNet().to(DEVICE)
trainloader, valloader, testloader = load_partition(partition_id=partition_id, batch_size=16)



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, valloader, num_epochs=10)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)