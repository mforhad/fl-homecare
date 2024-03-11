import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


# remove this class
class Net(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1800, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Working class. rename it to Lenet
class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        #print(x.shape)
        x = torch.reshape(x, (-1, 900, 2))
        #print(x.shape)
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


def train_gpt_working2(model, trainloader, valloader, optimizer, num_epochs=10, device="cuda"):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels.long())  # Ensure labels are of type torch.long
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Free up GPU memory
            del inputs, labels, outputs
            torch.cuda.empty_cache()

        train_loss = running_loss / len(trainloader)
        train_accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {train_loss:.4f}, "
              f"Accuracy: {100 * train_accuracy:.2f}%")

    print("Finished Training")

def train_gpt_working(model, trainloader, valloader, optimizer, num_epochs=10, device="cuda"):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape outputs if needed
            # outputs = torch.reshape(outputs, (-1,))

            loss = criterion(outputs, labels.long())  # Ensure labels are of type torch.long
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {train_loss:.4f}, "
              f"Accuracy: {100 * train_accuracy:.2f}%")

    print("Finished Training")

def train_has_error(model, train_loader, test_loader, optimizer, num_epochs, device: str):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch, 0.155))

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            outputs = torch.reshape(outputs, (-1,))
            labels = torch.repeat_interleave(labels, 2)
            # print("Output Shape:", outputs.shape)
            # print("Labels Shape:", labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # outputs = torch.reshape(outputs, (-1,))

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(test_loader)
            epoch_acc = correct / total

            print(f"Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")

        scheduler.step()


def train(model, train_loader, test_loader, optimizer, num_epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch, 0.155))

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs = torch.reshape(outputs, (-1,))
            # labels = torch.repeat_interleave(labels, 2)
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
                inputs, labels = inputs.to(device), labels.to(device)
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


# working runction. rename it to train
def train2(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            outputs = net(images)
            outputs = torch.reshape(outputs, (-1,))
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()


# Remove this function
def train3(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()


def test(model, test_loader, device):
    # Test model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

    # Save prediction scores
    y_score = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_score.extend(outputs.cpu().numpy()[:, 1])

    # output = pd.DataFrame({"y_true": y_test, "y_score": y_score, "subject": groups_test})
    # output.to_csv(os.path.join("output", "LeNet_pytorch.csv"), index=False)

    return 0.0, accuracy


# Working function. rename it to test
def test2(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            features, labels = data[0].to(device), data[1].to(device)  # move data to device
            outputs = net(features)  # forward pass
            outputs = torch.reshape(outputs, (-1,))
            #print(outputs.dtype)
            #print(labels.dtype)
            loss += criterion(outputs, labels).item()  # calculate loss
            predicted = (outputs > 0.5).float()  # convert probabilities to binary predictions
            correct += (predicted == labels).sum().item()  # count correct predictions

    test_loss = loss / len(testloader.dataset)
    test_accuracy = correct / len(testloader.dataset)
    #print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')
    return loss, test_accuracy

# remove this function
def test3(net, testloader, device: str):
    """Validate the network on the entire test set."""
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            # data = data.type(torch.LongTensor)  # casting to long
            features, labels = data[0], data[1]
            # features = features.type(torch.LongTensor)
            # labels = labels.type(torch.LongTensor)
            images, labels = features.to(device), labels.to(device)
            outputs = net(images)
            outputs = torch.reshape(outputs,(-1,))
            #print(outputs)
            #print(labels)
            #print(outputs.shape)
            #print(labels.shape)
            loss += criterion(outputs, labels).item()
            #_, predicted = torch.max(outputs.data, 1)
            predicted = outputs
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def get_mnist(data_path: str = "./data"):
    """Download MNIST and apply transform."""

    # transformation to convert images to tensors and apply normalization
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # prepare train and test set
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


