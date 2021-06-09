import os
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy
from torchvision.io import read_image
from torchvision.models import resnet18
from torchvision.transforms import Normalize


class FacialAgeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, download: Optional[bool] = None):
        if download is None:
            download = not os.path.exists(os.path.join(path, "face_age"))

        if download:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            os.makedirs(path, exist_ok=True)
            api.dataset_download_files(
                "frabbisw/facial-age", path=path, quiet=False, unzip=True
            )
        path = os.path.join(path, "face_age")

        self.path = path
        self.class_mapping = {}
        self.files = []

        # imagenet statistics
        self.normalization = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        for idx, directory in enumerate(
            sorted(
                [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
            )
        ):

            # skip directories that don't represent a class
            try:
                int(directory)
            except (TypeError, ValueError):
                continue

            self.class_mapping[int(directory)] = idx
            for file in sorted(
                [
                    x
                    for x in os.listdir(os.path.join(path, directory))
                    if os.path.isfile(os.path.join(path, directory, x))
                    and x.endswith(".png")
                ]
            ):
                self.files.append((directory, file))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        directory, file = self.files[index]
        label = torch.tensor(self.class_mapping[int(directory)]).long()

        image = read_image(os.path.join(self.path, directory, file))

        # convert from [0, 255] to [0, 1] since Normalize expects that range
        image = image / 255.0

        return self.normalization(image), label

    def __len__(self) -> int:
        return len(self.files)


class LitModule(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # load (pretrained) model
        self.model = resnet18(pretrained=pretrained)
        # change number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        # have different metrics for train and val to not mix up their internal state
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        y_hat = self.model(x)

        loss_val = self.loss_fn(y_hat, y)

        self.train_acc(y_hat.softmax(1), y)
        self.log("train_acc", self.train_acc)
        self.log("train_loss", loss_val)

        return loss_val

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch

        y_hat = self.model(x)

        loss_val = self.loss_fn(y_hat, y)

        self.val_acc(y_hat.softmax(1), y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss_val)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters())


def get_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=128)


if __name__ == "__main__":
    # seed for reproducability
    torch.manual_seed(42)
    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    dataset = FacialAgeClassificationDataset("/tmp/data")
    trainer = pl.Trainer(gpus=gpus)
    model = LitModule(num_classes=len(dataset.class_mapping))

    # use 75% of the data for training, rest for validation
    trainset_length = int(len(dataset) * 0.75)
    valset_length = len(dataset) - trainset_length

    trainset, validationset = torch.utils.data.random_split(
        dataset, [trainset_length, valset_length]
    )

    trainer.fit(model, get_dataloader(trainset), get_dataloader(validationset))

    # split trainset into 2 parts
    trainset_part1, trainset_part2 = torch.utils.data.random_split(
        trainset, [trainset_length // 2, trainset_length - trainset_length // 2]
    )

    # recreate model
    model = LitModule(num_classes=len(dataset.class_mapping))

    # merge back datasets
    trainset = torch.utils.data.ConcatDataset([trainset_part1, trainset_part2])
    trainer.fit(model, get_dataloader(trainset), get_dataloader(validationset))
