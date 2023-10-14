import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AutoEye(nn.Module):
    """AutoEye Model for classifying images as fictive or correct"""

    def __init__(self) -> None:
        """Model initialization"""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference of the model"""
        return x

    def get_parameters_amount(self) -> int:
        """Returns number of parameters of the Model"""
        return sum(p.numel() for p in self.parameters())
