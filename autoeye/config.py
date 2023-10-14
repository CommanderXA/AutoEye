from logging import Logger

import torch
from omegaconf import DictConfig


class Config:
    cfg = None
    device = torch.device("cpu")
    log = None
    training_model_name = None
    trained_epochs = 0
    train = False

    @classmethod
    def setup(cls, cfg: DictConfig, log: Logger, train: bool = False) -> None:
        cls.set_cfg(cfg)
        cls.set_device()
        cls.set_log(log)
        if train:
            cls.set_train()

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

    @classmethod
    def set_train(cls) -> None:
        cls.train = True
