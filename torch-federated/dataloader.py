import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np

from aws_s3_bucket import load_data_from_s3


DATA_FROM_S3_BUCKET = True
NUM_CLIENTS = 2


def load_numpy_tensor(file_name):
    file_path = "data/sleepapnea" + file_name + ".npy"
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


def load_partition(partition_id, batch_size=16, val_ratio=0.1):
    """Download and partitions the MNIST dataset."""

    if DATA_FROM_S3_BUCKET:
        trainset, testset = load_data_from_s3()
    else:
        trainset, testset = data_loader()

    # Split trainset into `num_partitions` trainsets
    data_points_per_client = len(trainset) // NUM_CLIENTS
    partition_len = [data_points_per_client] * NUM_CLIENTS
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

    # return train_partitions[partition_id], val_partitions[partition_id], test_partition[partition_id]
    return trainloader, valloader, testloader