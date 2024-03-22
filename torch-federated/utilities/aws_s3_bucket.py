import boto3
import numpy as np
import os

import torch
import torch.utils.data as data_utils

from io import BytesIO
from .utils import read_s3_secret_keys

# Replace these values with your own 
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = read_s3_secret_keys()

AWS_REGION = 'eu-central-1'
BUCKET_NAME = 'flhomecare'
UPLOAD_DATA = False

S3_KEYS = ['sleepapneatrain_features.npy', 'sleepapneatest_features.npy', 'sleepapneatrain_labels.npy', 'sleepapneatest_labels.npy']


s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION)

def upload_npy_to_s3(file_path, bucket_name, s3_key):
    s3.Bucket(bucket_name).upload_file(file_path, s3_key)

if UPLOAD_DATA:
    for file_name in S3_KEYS:
        file_path = os.path.join("/home/forhad/thesis/projects/fl-homecare/data", file_name)

        upload_npy_to_s3(file_path, BUCKET_NAME, file_name)


def load_npy_from_s3(s3_key):
    obj = s3.Object(BUCKET_NAME, s3_key)
    body = obj.get()['Body'].read()
    np_data = np.load(BytesIO(body))
    return np_data


def load_data_from_s3():
    train_features = torch.tensor(load_npy_from_s3(S3_KEYS[0]))
    test_features = torch.tensor(load_npy_from_s3(S3_KEYS[1]))
    train_labels = torch.tensor(load_npy_from_s3(S3_KEYS[2]))
    test_labels = torch.tensor(load_npy_from_s3(S3_KEYS[3]))

    train_features = torch.reshape(train_features, (train_features.shape[0], train_features.shape[1] * train_features.shape[2]))
    test_features = torch.reshape(test_features, (test_features.shape[0], test_features.shape[1] * test_features.shape[2]))

    train_features = train_features[0:16700, :]
    train_labels = train_labels[0:16700]
    test_features = test_features[0:16700, :]
    test_labels = test_labels[0:16700]

    print(train_features.shape)
    print(train_labels.shape)
    print(test_features.shape)
    print(test_labels.shape)

    train_datasets = data_utils.TensorDataset(train_features, train_labels)
    test_datasets = data_utils.TensorDataset(test_features, test_labels)


    return train_datasets, test_datasets

