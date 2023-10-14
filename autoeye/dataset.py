import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import pandas as pd
from PIL import Image

from autoeye.config import Config


class AutoDataset(Dataset):
    def __init__(self, annotation_file: str) -> None:
        super().__init__()
        self.data = pd.read_csv(annotation_file, header=True, delimiter=";")
        self.transforms = v2.Compose(
            [
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data = self.data[index]
        image_path = data[1]
        image = Image.open(image_path.item()).convert("RGB")
        if Config.train:
            image = self.transforms(image)
        target = data[2]
        sub_target = data[3]
        return image, target, sub_target
