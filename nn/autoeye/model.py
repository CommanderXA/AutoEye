import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d

from .config import Config


# class Classifier(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.fc1 = None
#         if Config.cfg.model.backbone == "resnet":
#             self.fc1 = nn.Linear(2048, 256)
#         else:
#             self.fc1 = nn.Linear(384, 256)
#         self.fc2 = nn.Linear(256, 5)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.relu(self.fc1(x))
#         x = F.softmax(self.fc2(x), dim=1)
#         return x

class DinoVision(nn.Module):
    def __init__(self):
        super(DinoVision, self).__init__()
        self.transformer = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.cls = pickle.load("./models/bag_lc_l_0964.pickle")

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.cls(x)
        return x


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = None
        if Config.cfg.model.backbone == "resnet":
            self.fc1 = nn.Linear(2048, 5)
        else:
            self.fc1 = nn.Linear(384, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        return x


class AutoEye(nn.Module):
    """AutoEye Model for classifying images as fictive or correct"""

    def __init__(self) -> None:
        """Model initialization"""
        super().__init__()
        self.backbone = None
        self.classifier = None
        if Config.cfg.model.backbone == "dino_vision":
            self.classifier = pickle.load("./models/bag_lc_l_0964.pickle")
        else:
            self.classifier = Classifier()
        self.__load_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference of the model"""
        x = self.backbone(x)
        if Config.cfg.model.backbone == "resnet":
            x = x.reshape(-1, 2048)
        else:
            x = self.backbone.norm(x)
        x = self.classifier(x)
        return x

    def __load_backbone(self) -> None:
        """Loads backbone model"""
        if Config.cfg.model.backbone == "resnet":
            self.backbone = resnext50_32x4d()
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif Config.cfg.model.backbone == "dino_vision":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.__freeze_backbone()

    def load(self, checkpoint) -> None:
        """Loads the model"""
        if Config.cfg.model.backbone != "dino_vision":
            if Config.cfg.model.backbone == "resnet":
                self.backbone.load_state_dict(checkpoint["backbone"])
            self.classifier.load_state_dict(checkpoint["classifier"])
            Config.set_trained_epochs(checkpoint["epochs"])

    def get_parameters_amount(self) -> int:
        """Returns number of parameters of the Model"""
        return sum(p.numel() for p in self.parameters())

    def __freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
