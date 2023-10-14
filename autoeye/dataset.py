import torch
from torch.utils.data import Dataset

class AutoDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> torch.Tensor:
        pass
