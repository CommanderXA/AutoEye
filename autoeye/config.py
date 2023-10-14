from logging import Logger

import torch
from omegaconf import DictConfig


class Config:
    cfg = None
    device = torch.device("cpu")
    log = None
    training_model_name = None
    trained_epochs = 0

    @classmethod
    def setup(cls, cfg: DictConfig, log: Logger) -> None:
        cls.set_cfg(cfg)
        cls.set_device()
        cls.set_log(log)
        # cls.set_current_model_name(
        #     name=f"model_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.pt")

    @classmethod
    def set_cfg(cls, cfg: DictConfig) -> None:
        cls.cfg = cfg

    @classmethod
    def set_device(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def set_log(cls, log) -> None:
        cls.log = log

    @classmethod
    def set_current_model_name(cls, name: str) -> None:
        cls.training_model_name = name

    @classmethod
    def set_trained_epochs(cls, epochs: int) -> None:
        cls.trained_epochs = epochs

    @classmethod
    def get_trained_epochs(cls) -> int:
        return cls.trained_epochs
