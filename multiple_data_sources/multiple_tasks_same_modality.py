import os
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy
from torchmetrics.regression import PearsonCorrcoef
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


class FacialAgeRegressionDataset(FacialAgeClassificationDataset):
    def __init__(self, path: str, download: Optional[bool] = None):
        super().__init__(path, download)

        # add inverse class mapping to reuse most of the code
        self.inverse_class_mapping = {v: k for k, v in self.class_mapping.items()}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = super().__getitem__(index)

        # revert back to original setting
        label = torch.tensor(self.inverse_class_mapping[label.item()]).float()

        return image, label


class NoOp(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LitModule(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # load (pretrained) model
        self.model = resnet18(pretrained=pretrained)

        # create different heads for classification and regression
        self.classification_head = torch.nn.Linear(
            self.model.fc.in_features, num_classes
        )
        self.regression_head = torch.nn.Linear(self.model.fc.in_features, 1)

        # replace linear layer by noop to only receive features
        self.model.fc = NoOp()

        # have different metrics for train and val to not mix up their internal state
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_pearson = PearsonCorrcoef()
        self.val_pearson = PearsonCorrcoef()

        self.loss_fn_classification = torch.nn.CrossEntropyLoss()
        self.loss_fn_regression = torch.nn.L1Loss()

    def training_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        batch_clf, batch_reg = batch

        x, y = batch_clf

        features = self.model(x)
        y_hat = self.classification_head(features)

        loss_val_clf = self.loss_fn_classification(y_hat, y)

        self.train_acc(y_hat.softmax(1), y)
        self.log("train_acc", self.train_acc)
        self.log("train_ce", loss_val_clf)

        x, y = batch_reg
        features = self.model(x)
        y_hat = self.regression_head(features).view_as(y)

        loss_val_reg = self.loss_fn_regression(y_hat, y)
        self.train_pearson(y_hat, y)
        self.log("train_pearson", self.train_pearson)
        self.log("train_l1", loss_val_reg)

        loss_val = loss_val_clf + loss_val_reg
        self.log("train_loss_total", loss_val)

        return loss_val

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = batch

        features = self.model(x)

        # classification case
        if dataloader_idx == 0:
            y_hat = self.classification_head(features)
            loss_val = self.loss_fn_classification(y_hat, y)

            self.val_acc(y_hat.softmax(1), y)
            self.log("val_acc", self.val_acc)
            self.log("val_ce", loss_val)

        elif dataloader_idx == 1:
            y_hat = self.regression_head(features).view_as(y)
            loss_val = self.loss_fn_regression(y_hat, y)
            self.val_pearson(y_hat, y)
            self.log("val_pearson", self.val_pearson)
            self.log("val_l1", loss_val)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())


def get_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=128)


if __name__ == "__main__":
    # seed for reproducability
    torch.manual_seed(42)
    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    dataset_classification = FacialAgeClassificationDataset("/tmp/data")
    dataset_regression = FacialAgeRegressionDataset("/tmp/data")
    trainer = pl.Trainer(gpus=gpus)
    model = LitModule(num_classes=len(dataset_classification.class_mapping))

    # use 75% of the data for training, rest for validation
    trainset_clf_length = int(len(dataset_classification) * 0.75)
    valset_clf_length = len(dataset_classification) - trainset_clf_length
    trainset_reg_length = int(len(dataset_regression) * 0.75)
    valset_reg_length = len(dataset_regression) - trainset_reg_length

    trainset_clf, validationset_clf = torch.utils.data.random_split(
        dataset_classification, [trainset_clf_length, valset_clf_length]
    )
    trainset_reg, validationset_reg = torch.utils.data.random_split(
        dataset_regression, [trainset_reg_length, valset_reg_length]
    )

    # those will create a single batch consisting of two subbatches: one from the clf loader and one from the reg loader
    trainloaders = [get_dataloader(trainset_clf), get_dataloader(trainset_reg)]

    # they will be used sequentially
    valloaders = [get_dataloader(validationset_clf), get_dataloader(validationset_reg)]

    trainer.fit(model, trainloaders, valloaders)
