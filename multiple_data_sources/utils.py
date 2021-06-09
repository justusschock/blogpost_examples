import os
import torch

def get_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=128)

def download_if_necessary(path, download, dataset, path_ext):
    if download is None:
            download = not os.path.exists(os.path.join(path, path_ext))

    if download:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        os.makedirs(path, exist_ok=True)
        api.dataset_download_files(
            dataset, path=path, quiet=False, unzip=True
        )
    path = os.path.join(path, path_ext)

    return path

def split_data(dataset, train_percentage, val_percentage=None):
    trainset_length = int(len(dataset) * train_percentage)
    if val_percentage is None:
        valset_length = len(dataset) - len(trainset_length)
    else:
        valset_length = int(len(dataset) * val_percentage)
    
    trainset, validationset = torch.utils.data.random_split(
        dataset, [trainset_length, valset_length]
    )

    return trainset, validationset