import torch
import torch.nn as nn
import torch.optim as optim

# import psutil
import time
import pyRAPL

from data.dataloader import fl_config
from utilities.gpu_energy_metric import get_gpu_energy_consumption

if fl_config.should_use_gpu:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")

consumed_energy = 0

try:
    pyRAPL.setup()
    meter = pyRAPL.Measurement('training_energy_consumption')
except Exception as e:
    print(f"Error: {e}")
    print("pyRAPL does not work on this client!!!")
    consumed_energy = 0

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
    # print("Learning rate: ", lr)
    return lr


def train(client_id, model, train_loader, test_loader, num_epochs):
    initial_time = time.time()
    try:
        # initial_power = psutil.sensors_battery().percent
        meter.begin()
    except Exception:
        consumed_energy = 0
        pass

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch, fl_config.lr))

    model.to(DEVICE)
    model.train()
    for epoch in range(num_epochs):
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

            print(f"Client# {client_id}:Epoch {epoch + 1}/{num_epochs}, Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")

        scheduler.step()

    try:
        meter.end()

        dram_energy = meter.result.dram[0]
        pkg_energy = meter.result.pkg[0]

        print(f"Client# {client_id}: Training Energy consumed (result): {meter.result}") 
        print(f"Client# {client_id}: Training Energy consumed (pkg): {(pkg_energy / 1e6):.2f} J")
        print(f"Client# {client_id}: Training Energy consumed (dram): {(dram_energy / 1e6):.2f} J")
        print(f"Client# {client_id}: Training time: {meter.result.duration}")

        gpu_energy_use = get_gpu_energy_consumption(meter.result.duration)
        print(f"Client# {client_id}: GPU Energy consumed: {(gpu_energy_use / 1e3):.2f}")
        consumed_energy = (pkg_energy / 1e9) + (dram_energy / 1e9) + (gpu_energy_use / 1e3)
        print(f"Client# {client_id}: Training Energy consumed (total): {consumed_energy:.2f} kJ")

        # final_power = psutil.sensors_battery().percent
        # energy_consumed = initial_power - final_power
        # print(f"Client# {client_id}: Energy consumed: {energy_consumed:.4f}")

    except Exception as e:
        print("Energy consumption cannot be determined for this device")
        consumed_energy = 0

    final_time = time.time()
    training_time = final_time - initial_time
    print(f"Client# {client_id}: Training time: {training_time}")

    return training_time, consumed_energy


def test(client_id, model, test_loader, y_test=None, groups_test=None):
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
        print(f"Client# {client_id}: Test Accuracy: {accuracy:.4f}")

    return loss, accuracy