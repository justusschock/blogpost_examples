import os
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy
from torchmetrics.regression import PearsonCorrcoef
from torchvision.io import read_image
from torchvision.models import resnet18
from torchvision.transforms import Normalize

from .utils import get_dataloader, download_dataset_if_necessary, split_data

# 1.) Define First Task Dataset
class FacialAgeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, download: Optional[bool] = None):
        path = download_dataset_if_necessary(path, download, "frabbisw/facial-age", "face_age")

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

            # map to continuous class indices
            self.class_mapping[int(directory)] = idx

            # parse all files
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
        # get directory and file
        directory, file = self.files[index]

        # get label (from directory name)
        label = torch.tensor(self.class_mapping[int(directory)]).long()

        # read image
        image = read_image(os.path.join(self.path, directory, file))

        # convert from [0, 255] to [0, 1] since Normalize expects that range
        image = image / 255.0

        # apply normalization to image
        return self.normalization(image), label

    def __len__(self) -> int:
        return len(self.files)


# 2.) Define Second Task Dataset
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

# 3.) Define Model
class LitModule(pl.LightningModule):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()

        # load (pretrained) model
        self.model = resnet18(pretrained=pretrained)

        # 4.) Create different heads for each Task
        self.classification_head = torch.nn.Linear(
            self.model.fc.in_features, num_classes
        )
        self.regression_head = torch.nn.Linear(self.model.fc.in_features, 1)

        # replace linear layer by noop to only receive features
        self.model.fc = torch.nn.Identity()

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

        # 5.) In training: Split into sub batches
        batch_clf, batch_reg = batch


        # 6.) Do Forward and metric calculation of one task
        x, y = batch_clf

        features = self.model(x)
        y_hat = self.classification_head(features)

        loss_val_clf = self.loss_fn_classification(y_hat, y)

        self.train_acc(y_hat.softmax(1), y)
        self.log("train_acc", self.train_acc)
        self.log("train_ce", loss_val_clf)

        # 7.) Do forward and metric calculation of other task
        x, y = batch_reg
        features = self.model(x)
        y_hat = self.regression_head(features).view_as(y)

        loss_val_reg = self.loss_fn_regression(y_hat, y)
        self.train_pearson(y_hat, y)
        self.log("train_pearson", self.train_pearson)
        self.log("train_l1", loss_val_reg)

        # 8.) Combine them (here by loss summation)
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

        # 9.) In validation: Datasets are sampled sequentially. Switch Task depending on dataloader_idx
        if dataloader_idx == 0:
            # 10.) Implement Validation for first task
            # classification case
            y_hat = self.classification_head(features)
            loss_val = self.loss_fn_classification(y_hat, y)

            self.val_acc(y_hat.softmax(1), y)
            self.log("val_acc", self.val_acc)
            self.log("val_ce", loss_val)

        elif dataloader_idx == 1:
            # 11.) Implement Validation for second Task
            y_hat = self.regression_head(features).view_as(y)
            loss_val = self.loss_fn_regression(y_hat, y)
            self.val_pearson(y_hat, y)
            self.log("val_pearson", self.val_pearson)
            self.log("val_l1", loss_val)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    # 12.) seed for reproducability
    torch.manual_seed(42)

    # 13.) Select Devices for training
    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    # 14.) Instantiate datasets
    dataset_classification = FacialAgeClassificationDataset("/tmp/data")
    dataset_regression = FacialAgeRegressionDataset("/tmp/data")

    # 14.1) Optionally split your data
    # use 75% of the data for training, rest for validation
    trainset_reg, validationset_reg = split_data(dataset_regression, 0.75)
    trainset_clf, validationset_clf = split_data(dataset_classification, 0.75)

    # 15.) Create DataLoaders
    # those will create a single batch consisting of two subbatches: one from the clf loader and one from the reg loader
    trainloaders = [get_dataloader(trainset_clf, True), get_dataloader(trainset_reg, True)]

    # they will be used sequentially
    valloaders = [get_dataloader(validationset_clf, False), get_dataloader(validationset_reg, False)]

    # 16.) Create Trainer and Model
    trainer = pl.Trainer(gpus=gpus)
    model = LitModule(num_classes=len(dataset_classification.class_mapping))

    # 17.) Train!
    trainer.fit(model, trainloaders, valloaders)
