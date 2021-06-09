import os
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy
from torchvision.io import read_image
from torchvision.models import resnet50
from torchvision.transforms import Normalize, Resize


class MelanomaImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, root_dir: str) -> None:
        self.annotations = pd.read_csv(csv_file).dropna()
        self.root_dir = root_dir

        # use imagenet statistics
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize = Resize((224, 224))

    def __getitem__(self, index: int) -> torch.Tensor:

        image = read_image(
            os.path.join(self.root_dir, self.annotations.iloc[index, 0] + ".jpg")
        )

        # convert from [0, 255] to [0, 1] since Normalize expects that range
        image = image / 255.0

        image = self.resize(image)

        return self.normalize(image)

    def __len__(self) -> int:
        return len(self.annotations)


class MelanomaCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str) -> None:
        self.annotations = pd.read_csv(csv_file).dropna()
        self.sex_class_embedding = {"male": 0.0, "female": 1.0}

    def __getitem__(self, index: int) -> Tuple[float, float, int]:
        sex = self.sex_class_embedding[self.annotations.iloc[index, 2]]
        age = float(self.annotations.iloc[index, 3])
        target = int(self.annotations.iloc[index, 7])

        return sex, age, target

    def __len__(self) -> int:
        return len(self.annotations)


class CombinedMelanomaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, root_dir: str) -> None:
        self.image_dataset = MelanomaImageDataset(csv_file, root_dir)
        self.csv_dataset = MelanomaCSVDataset(csv_file)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float, float, int]:
        image_sample = self.image_dataset[index]
        csv_sample = self.csv_dataset[index]

        return (image_sample, *csv_sample)

    def __len__(self) -> int:
        return min(len(self.image_dataset), len(self.csv_dataset))


class MyLitModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.image_model = resnet50(pretrained=True)
        self.image_model.fc = torch.nn.Linear(self.image_model.fc.in_features, 2)
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.acc_train = Accuracy()
        self.acc_val = Accuracy()

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x_img, x_sex, x_age, y = batch
        x_sex, x_age = x_sex.float(), x_age.float()

        y_img = self.image_model(x_img)

        y_hat = self.mlp_head(
            torch.cat([y_img, x_sex.unsqueeze(1), x_age.unsqueeze(1)], dim=-1)
        )
        loss_val = self.loss_fn(y_hat, y)

        self.acc_train(y_hat.softmax(1), y)
        self.log("train/acc", self.acc_train)
        self.log("train/ce", loss_val)

        return loss_val

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x_img, x_sex, x_age, y = batch
        x_sex, x_age = x_sex.float(), x_age.float()

        y_img = self.image_model(x_img)

        y_hat = self.mlp_head(
            torch.cat([y_img, x_sex.unsqueeze(1), x_age.unsqueeze(1)], dim=-1)
        )
        loss_val = self.loss_fn(y_hat, y)

        self.acc_val(y_hat.softmax(1), y)
        self.log("val/acc", self.acc_val)
        self.log("val/ce", loss_val)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    torch.manual_seed(42)

    dataset_path = "/tmp/data/kaggle"

    if not os.path.exists(dataset_path) or not os.path.exists(
        os.path.join(dataset_path, "siim-isic-melanoma-classification")
    ):
        import zipfile

        from kaggle.api.kaggle_api_extended import KaggleApi
        from tqdm import tqdm

        api = KaggleApi()
        api.authenticate()

        os.makedirs(dataset_path, exist_ok=True)

        zip_path = os.path.join(dataset_path, "siim-isic-melanoma-classification.zip")
        if not os.path.isfile(zip_path):
            api.competition_download_files(
                "siim-isic-melanoma-classification", path=dataset_path, quiet=False
            )

        print("Extract Zipfile")
        with zipfile.ZipFile(zip_path) as f:
            unzip_path = zip_path.rsplit(".", 1)[0]
            for file in tqdm(f.infolist()):
                f.extract(file, unzip_path)
        os.remove(zip_path)

    dataset_path = os.path.join(dataset_path, "siim-isic-melanoma-classification")

    dataset = CombinedMelanomaDataset(
        csv_file=os.path.join(dataset_path, "train.csv"),
        root_dir=os.path.join(dataset_path, "jpeg", "train"),
    )

    model = MyLitModel()

    gpus = None
    if torch.cuda.is_available():
        gpus = 1

    trainset, valset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.75), len(dataset) - int(len(dataset) * 0.75)]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, num_workers=8, pin_memory=True, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=64, num_workers=8, pin_memory=True, shuffle=False
    )

    trainer = pl.Trainer(gpus=gpus)
    trainer.fit(model, trainloader, valloader)
