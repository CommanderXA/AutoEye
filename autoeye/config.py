from logging import Logger

import torch

from omegaconf import DictConfig


class Config:
    log = None
    cfg: DictConfig = None
    model_path: str = None
    device = torch.device("cpu")
    trained_epochs: int = 0
    train: bool = False
    best_accuracy: float = 0.0

    @classmethod
    def setup(cls, cfg: DictConfig, log: Logger, train: bool = False) -> None:
        cls.set_cfg(cfg)
        cls.set_device()
        cls.set_log(log)
        cls.set_current_model_path()
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
    def set_current_model_path(cls) -> None:
        name = cls.cfg.model.name
        backbone = cls.cfg.model.backbone
        cls.model_path = f"models/{name}_{backbone}.pt"

    @classmethod
    def set_trained_epochs(cls, epochs: int) -> None:
        cls.trained_epochs = epochs

    @classmethod
    def get_trained_epochs(cls) -> int:
        return cls.trained_epochs

    @classmethod
    def set_train(cls) -> None:
        cls.train = True

    @classmethod
    def set_best_accuracy(cls, accuracy: float) -> None:
        cls.best_accuracy = accuracy
