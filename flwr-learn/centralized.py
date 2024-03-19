import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import pandas as pd
import os

from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

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


def train(model, train_loader, val_loader, num_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.155, momentum=0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch, 0.01))

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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)

                loss = criterion(outputs, labels.long())
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Free up GPU memory
                torch.cuda.empty_cache()

            epoch_loss = running_loss / len(val_loader)
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
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                y_score.extend(outputs.cpu().numpy()[:, 1])

        output = pd.DataFrame({"y_true": y_test, "y_score": y_score, "subject": groups_test})
        output.to_csv(os.path.join("output", "homecare_pytorch.csv"), index=False)

    return loss, accuracy


def load_numpy_tensor(file_name):
    file_path = "../data/sleepapnea" + file_name + ".npy"
    return np.load(file_path)


def load_data():
    train_features = torch.tensor(load_numpy_tensor("train_features"))
    test_features = torch.tensor(load_numpy_tensor("test_features"))
    train_labels = torch.tensor(load_numpy_tensor("train_labels"))
    test_labels = torch.tensor(load_numpy_tensor("test_labels"))

    train_features = torch.reshape(train_features, (train_features.shape[0], train_features.shape[1] * train_features.shape[2]))
    test_features = torch.reshape(test_features, (test_features.shape[0], test_features.shape[1] * test_features.shape[2]))

    train_features = train_features[0:16700, :]
    train_labels = train_labels[0:16700]
    test_features = test_features[0:16700, :]
    test_labels = test_labels[0:16700]

    train_datasets = data_utils.TensorDataset(train_features, train_labels)
    test_datasets = data_utils.TensorDataset(test_features, test_labels)

    trainloader = data_utils.DataLoader(train_datasets, batch_size=16)
    testloader = data_utils.DataLoader(test_datasets, batch_size=16)

    # return train_datasets, test_datasets
    return trainloader, testloader


def load_model():
    return LeNet().to(DEVICE)


if __name__ == "__main__":
    net = load_model()
    trainloader, testloader = load_data()

    train(net, trainloader, testloader, num_epochs=15)

    loss, accuracy = test(net, test_loader=testloader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")