import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np

from .aws_s3_bucket import load_data_from_s3
from config.fl_config import FlHomecareConfig

fl_config = FlHomecareConfig()

def load_numpy_tensor(file_name):
    file_path = "datasets/sleepapnea" + file_name + ".npy"
    return np.load(file_path)


def data_loader():
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

    train_datasets = TensorDataset(train_features, train_labels)
    test_datasets = TensorDataset(test_features, test_labels)

    return train_datasets, test_datasets


def load_partition(partition_id, batch_size=fl_config.batch_size, val_ratio=fl_config.val_ratio):
    """Download and partitions the MNIST dataset."""

    if fl_config.is_data_from_s3:
        trainset, testset = load_data_from_s3()
    else:
        trainset, testset = data_loader()

    # Split trainset into `num_partitions` trainsets
    trainset_len = len(trainset)
    if fl_config.is_heterogeneous:
        partition_len = distribute_data_randomly(trainset_len, fl_config.num_clients)
    else:
        data_points_per_client = trainset_len // fl_config.num_clients
        partition_len = [data_points_per_client] * fl_config.num_clients
        print("data_points_per_client", data_points_per_client)

    print("partition_len", partition_len)

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2024)
    )

    test_partition = random_split(
        testset, partition_len, torch.Generator().manual_seed(2024)
    )

    # Create dataloaders with train+val support
    train_partitions = []
    val_partitions = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2024)
        )

        train_partitions.append(for_train)
        val_partitions.append(for_val)

    trainloader = DataLoader(train_partitions[partition_id], batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_partitions[partition_id], batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_partition[partition_id], batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader

def distribute_data_randomly(data_size, num_clients, seed=2024):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    parts = torch.randint(1, data_size, (num_clients - 1,))    
    # Add endpoints to the random numbers
    parts = torch.cat([torch.tensor([0]), parts, torch.tensor([data_size])])    
    parts = torch.sort(parts).values
    
    # Calculate the lengths of partitions
    partition_lengths = parts[1:] - parts[:-1]
    
    return partition_lengths