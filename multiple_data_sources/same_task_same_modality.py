import os
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy
from torchvision.io import read_image
from torchvision.models import resnet18
from torchvision.transforms import Normalize

from .utils import download_dataset_if_necessary, get_dataloader, split_data


# 1.) Define Dataset
class FacialAgeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, download: Optional[bool] = None):
        path = download_dataset_if_necessary(
            path, download, "frabbisw/facial-age", "face_age"
        )

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


# 2.) Define Model
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
        # 3.) In training: Only one batch, always of same type
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
        # 4.) In validation: Only one batch, always of same type
        x, y = batch

        y_hat = self.model(x)

        loss_val = self.loss_fn(y_hat, y)

        self.val_acc(y_hat.softmax(1), y)
        self.log("val_acc", self.val_acc)
        self.log("val_loss", loss_val)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters())


if __name__ == "__main__":
    # 5.) seed for reproducability
    torch.manual_seed(42)

    # 6.) Select Devices for training
    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    # 7.) Instantiate Dataset
    dataset = FacialAgeClassificationDataset("/tmp/data")

    # ONLY HERE SINCE WE USE TWO PARTS OF THE SAME DATASET FOR DEMONSTRATION:
    # split into dataset parts
    dataset1, dataset2 = split_data(dataset, 0.5)

    # 7.1 Optionally: Split into trainset and validationset
    trainset1, valset1 = split_data(dataset1, 0.75)
    trainset2, valset2 = split_data(dataset2, 0.75)

    # 8.) Concatenate Datasets (they produce batches of the same type)
    trainset = torch.utils.data.ConcatDataset([trainset1, trainset2])
    valset = torch.utils.data.ConcatDataset([valset1, valset2])

    # 9.) Create dataloaders
    trainloader = get_dataloader(trainset, True)
    valloader = get_dataloader(valset, False)

    # 10.) Create Model and Trainer
    trainer = pl.Trainer(gpus=gpus)
    model = LitModule(num_classes=len(dataset.class_mapping))

    # 11.) Train!
    trainer.fit(model, trainloader, valloader)
