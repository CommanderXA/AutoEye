import logging
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

import hydra
from omegaconf import DictConfig

from tqdm import tqdm

from autoeye.config import Config
from autoeye.dataset import AutoDataset
from autoeye.model import AutoEye
from utils import evaluate, evaluate_accuracy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log, train=True)

    if Config.cfg.hyper.use_amp:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        dtype = "bfloat16"
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # loading model
    model = AutoEye()
    # model = torch.compile(model)
    model.eval()

    if cfg.hyper.pretrained and os.path.exists(f"{Config.model_path[:-3]}.pt"):
        checkpoint = torch.load(f"{Config.model_path[:-3]}.pt")
        model.load(checkpoint)
        Config.set_trained_epochs(checkpoint["epochs"])

    model.to(device=Config.device)
    logging.info(
        f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {Config.get_trained_epochs()} epochs)"
    )

    # train dataset
    dataset = AutoDataset(cfg.data.csv_files[0])
    # train, validation dataset split
    _, validset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # validation loader
    validloader = DataLoader(
        dataset=validset, batch_size=cfg.hyper.batch_size, shuffle=True
    )
    # preform evaluation
    evaluate(model, validloader)


if __name__ == "__main__":
    # torch.manual_seed(42)
    # torch.multiprocessing.set_start_method("spawn")
    eval()
