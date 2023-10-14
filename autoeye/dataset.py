import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import pandas as pd
from PIL import Image

from autoeye.config import Config


class AutoDataset(Dataset):
    def __init__(self, annotation_file: str) -> None:
        super().__init__()
        self.data = pd.read_csv(annotation_file, header=0, index_col=0, delimiter=";")
        self.transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.train_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data = self.data.iloc[index]
        image_path = data[0]
        image = Image.open(image_path).convert("RGB")
        if Config.train:
            image = self.train_transforms(image)
        else:
            image = self.transforms(image)
        image = image.to(Config.device)
        target = torch.Tensor([data[1]]).to(Config.device)
        sub_target = torch.Tensor([data[2]]).to(Config.device)
        return image, target, sub_target
