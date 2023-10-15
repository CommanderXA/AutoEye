import pickle
import numpy as np
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
            with open("./models/classifier.pickle", "rb") as f:
                self.classifier = pickle.load(f)
        else:
            self.classifier = Classifier()
        self.__load_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference of the model"""
        x = self.backbone(x)
        if Config.cfg.model.backbone == "resnet":
            x = x.reshape(-1, 2048)
        else:
            if Config.cfg.model.backbone == "dino_vision":
                _x: list[torch.Tensor] = []
                for i in range(x.size(0)):
                    prediction = self.classifier.predict(
                        x[i].cpu().detach().numpy().reshape(1, -1)
                    )
                    if prediction == "0-correct":
                        prediction = 0
                    elif prediction == "1-not-on-the-brake-stand":
                        prediction = 1
                    elif prediction == "2-from-the-screen":
                        prediction = 2
                    elif prediction == "3-from-the-screen+photoshop":
                        prediction = 3
                    else:
                        prediction = 4

                    classes = torch.nn.functional.one_hot(torch.arange(0, 5)).float()
                    prediction = classes[prediction].to(Config.device).unsqueeze(0)
                    _x.append(prediction)
                x = torch.cat(_x, 0).to(Config.device)
                return x

        x = self.classifier(x)
        return x

    def __load_backbone(self) -> None:
        """Loads backbone model"""
        if Config.cfg.model.backbone == "resnet":
            self.backbone = resnext50_32x4d()
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif Config.cfg.model.backbone == "dino_vision":
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_lc"
            )
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
