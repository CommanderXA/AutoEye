import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms as T

import pandas as pd
from PIL import Image

from autoeye.config import Config


class AutoDataset(Dataset):
    def __init__(self, annotation_file: str) -> None:
        super().__init__()
        index_col = None
        if Config.train:
            index_col = 0
        self.data = pd.read_csv(
            annotation_file, header=0, index_col=index_col, delimiter=";"
        )
        self.transforms = None
        self.train_transforms = None
        if Config.cfg.model.backbone != "dino_vision":
            self.transforms = v2.Compose(
                [
                    v2.Resize(size=(224, 224), antialias=True),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.train_transforms = v2.Compose(
                [
                    v2.Resize(size=(224, 224), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation((0, 10)),
                    v2.ColorJitter(
                        brightness=(0.0, 0.2),
                        contrast=(0.0, 0.2),
                        saturation=(0.0, 0.2),
                        hue=0.0,
                    ),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transforms = T.Compose([T.ToTensor(), T.Resize(224), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
            self.train_transforms = T.Compose([T.ToTensor(), T.Resize(224), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data = self.data.iloc[index]
        if Config.train:
            image_path = data.iloc[0]
            image = Image.open(image_path).convert("RGB")
            image = self.train_transforms(image)
            image = image.to(Config.device)
            classes = torch.nn.functional.one_hot(torch.arange(0, 5)).float()
            target = data.iloc[2]
            # if Config.cfg.model.backbone == "dino_vision":
            #     if target > 0:
            #         target = 1
            target = classes[target].to(Config.device)
            return image, target

        image_path = (
            "./data/case3-datasaur-photo/techosmotr/techosmotr/test/"
            + str(data.iloc[0].item())
            + ".jpeg"
        )
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        image = image.to(Config.device)
        idx = torch.tensor([data.iloc[0]], dtype=torch.int64)
        idx = idx.to(Config.device)
        return idx, image
