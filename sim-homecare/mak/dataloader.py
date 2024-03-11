import pickle
import numpy as np
import os
from scipy.interpolate import splev, splrep
import torch.utils.data as data_utils
import torch

def save_numpy_tensor(numpy_tensor, file_name):
    save_file_path = "/home/forhad/thesis/projects/fl-homecare/sim-homecare/data/sleepapnea" + file_name + ".npy"

    with open(save_file_path, "wb") as npf:
        np.save(npf, numpy_tensor)


def load_numpy_tensor(file_name):
    save_file_path = "/home/forhad/thesis/projects/fl-homecare/sim-homecare/data/sleepapnea" + file_name + ".npy"
    return np.load(save_file_path)


def load_data():
    base_dir = "/home/forhad/thesis/projects/fl-homecare/dataset"

    ir = 3  # interpolate interval
    before = 2
    after = 2

    # normalize
    scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f: # read preprocessing result
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1)) # convert to numpy format
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    save_numpy_tensor(x_train, "train_features")
    save_numpy_tensor(x_test, "test_features")
    save_numpy_tensor(y_train, "train_labels")
    save_numpy_tensor(y_test, "test_labels")

    # x_train (16709, 900, 2)
    # y_train (16709,)
    # x_test (16945, 900, 2)
    # y_test (16945,)

    return x_train, y_train, groups_train, x_test, y_test, groups_test

# load_data()


def data_loader():
    train_features = torch.tensor(load_numpy_tensor("train_features"))
    test_features = torch.tensor(load_numpy_tensor("test_features"))
    train_labels = torch.tensor(load_numpy_tensor("train_labels"))
    test_labels = torch.tensor(load_numpy_tensor("test_labels"))

    train_features = torch.reshape(train_features, (train_features.shape[0], train_features.shape[1] * train_features.shape[2]))
    test_features = torch.reshape(test_features, (test_features.shape[0], test_features.shape[1] * test_features.shape[2]))

    # workaround for the following error
    #  File "/home/forhad/thesis/projects/fl-homecare/sim-homecare/simulation.py", line 197, in main
    # trainsets, valsets, testset = prepare_dataset()
    #   File "/home/forhad/thesis/projects/fl-homecare/sim-homecare/simulation.py", line 132, in prepare_dataset
    #     trainsets = random_split(
    #   File "/home/forhad/thesis/projects/fl-homecare/.venv/lib/python3.10/site-packages/torch/utils/data/dataset.py", line 454, in random_split
    #     raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    # ValueError: Sum of input lengths does not equal the length of the input dataset!

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

    # train_dataloader = data_utils.DataLoader(train_datasets)
    # test_dataloader = data_utils.DataLoader(test_datasets)

    return train_datasets, test_datasets

data_loader()