import os
import torch

def get_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=64)

def download_dataset_if_necessary(path, download, dataset, path_ext):
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

def download_competition_data_if_necessary(dataset_path, competition_name):
    if not os.path.exists(dataset_path) or not os.path.exists(
        os.path.join(dataset_path, competition_name)
    ):
        import zipfile

        from kaggle.api.kaggle_api_extended import KaggleApi
        from tqdm import tqdm

        api = KaggleApi()
        api.authenticate()

        os.makedirs(dataset_path, exist_ok=True)

        zip_path = os.path.join(dataset_path, f"{competition_name}.zip")
        if not os.path.isfile(zip_path):
            api.competition_download_files(
                competition_name, path=dataset_path, quiet=False
            )

        print("Extract Zipfile")
        with zipfile.ZipFile(zip_path) as f:
            unzip_path = zip_path.rsplit(".", 1)[0]
            for file in tqdm(f.infolist()):
                f.extract(file, unzip_path)
        os.remove(zip_path)

    dataset_path = os.path.join(dataset_path, competition_name)
    return dataset_path

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