from torch.utils.data import DataLoader


def read_s3_secret_keys():
    with open("secrets/accesskey.txt", "r") as f:
        access_key = f.read()

    with open("secrets/secretkey.txt", "r") as f:
        secret_key = f.read()

    return str(access_key), str(secret_key)

