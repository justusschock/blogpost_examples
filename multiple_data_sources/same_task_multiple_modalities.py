import os
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torchmetrics.classification import Accuracy
from torchvision.io import read_image
from torchvision.models import resnet50
from torchvision.transforms import Normalize, Resize

# 1.) Define Dataset for first modality
class MelanomaImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, root_dir: str) -> None:
        self.annotations = pd.read_csv(csv_file).dropna()
        self.root_dir = root_dir

        # use imagenet statistics
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.resize = Resize((224, 224))

    def __getitem__(self, index: int) -> torch.Tensor:

        # read image
        image = read_image(
            os.path.join(self.root_dir, self.annotations.iloc[index, 0] + ".jpg")
        )

        # convert from [0, 255] to [0, 1] since Normalize expects that range
        image = image / 255.0

        # resize image to target size
        image = self.resize(image)

        # normalize image
        return self.normalize(image)

    def __len__(self) -> int:
        return len(self.annotations)

# 2.) Define Dataset for second modality
class MelanomaCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str) -> None:
        self.annotations = pd.read_csv(csv_file).dropna()
        self.sex_class_embedding = {"male": 0.0, "female": 1.0}

    def __getitem__(self, index: int) -> Tuple[float, float, int]:
        # convert to categorical classes
        sex = self.sex_class_embedding[self.annotations.iloc[index, 2]]
        age = float(self.annotations.iloc[index, 3])
        target = int(self.annotations.iloc[index, 7])

        return sex, age, target

    def __len__(self) -> int:
        return len(self.annotations)

# 3.) Combine Datasets in separate Dataset class
class CombinedMelanomaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, root_dir: str) -> None:
        # instantiate sub datasets
        self.image_dataset = MelanomaImageDataset(csv_file, root_dir)
        self.csv_dataset = MelanomaCSVDataset(csv_file)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float, float, int]:
        # sample each of them
        image_sample = self.image_dataset[index]
        csv_sample = self.csv_dataset[index]

        # return combined sample
        return (image_sample, *csv_sample)

    def __len__(self) -> int:
        return min(len(self.image_dataset), len(self.csv_dataset))

# 4.) Define Model
class MyLitModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.image_model = resnet50(pretrained=True)
        self.image_model.fc = torch.nn.Linear(self.image_model.fc.in_features, 2)

        # outputs of image model will go into mlp head
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
        # 5.) In training: One batch containing all parts. Use them as desired
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
        # 6.) In validation: One batch containing all parts. Use them as desired
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
    from .utils import download_competition_data_if_necessary, split_data, get_dataloader

    # 7.) Seed for reproducability
    torch.manual_seed(42)

    # 8.) Select Devices for Training
    gpus = None
    if torch.cuda.is_available():
        gpus = 1

    # 9.) Instantiate (combined) Dataset
    dataset_path = "/tmp/data/kaggle"
    dataset_path = download_competition_data_if_necessary(dataset_path, "siim-isic-melanoma-classification")
    dataset = CombinedMelanomaDataset(
        csv_file=os.path.join(dataset_path, "train.csv"),
        root_dir=os.path.join(dataset_path, "jpeg", "train"),
    )

    # 9.1) Optionally: Split data into training and validationset
    trainset, valset = split_data(dataset, 0.75)

    # 10.) Create DataLoaders
    trainloader = get_dataloader(trainset, True)
    valloader = get_dataloader(valset, False)

    # 11.) Create Trainer and Model
    model = MyLitModel()
    trainer = pl.Trainer(gpus=gpus)

    # 12. Train
    trainer.fit(model, trainloader, valloader)
