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
def train(cfg: DictConfig) -> None:
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
    model = AutoEye().to(device=Config.device)
    model = torch.compile(model)

    # optimizers
    scaler = GradScaler(enabled=cfg.hyper.use_amp)
    optimizer = AdamW(
        model.parameters(), lr=cfg.hyper.lr, betas=(cfg.optim.beta1, cfg.optim.beta2)
    )

    if cfg.hyper.pretrained and os.path.exists(Config.model_path):
        checkpoint = torch.load(Config.model_path)
        model.load(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        Config.set_trained_epochs(checkpoint["epochs"])

    logging.info(
        f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {Config.get_trained_epochs()} epochs)"
    )

    # train dataset
    dataset = AutoDataset(cfg.data.csv_files[0])
    # train dataset split
    trainset, validset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # train dataloader
    trainloader = DataLoader(
        dataset=trainset, batch_size=cfg.hyper.batch_size, shuffle=True
    )
    validloader = DataLoader(
        dataset=validset, batch_size=cfg.hyper.batch_size, shuffle=True
    )
    # test dataset
    testdataset = AutoDataset(cfg.data.csv_files[1])
    # test dataloader
    testloader = DataLoader(
        dataset=testdataset, batch_size=cfg.hyper.batch_size, shuffle=False
    )

    logging.info(f"Training")
    # set mode of the model to train
    model.train()
    train_step(model, optimizer, scaler, trainloader, validloader)
    # test evaluation
    evaluate(model, validloader)


def train_step(
    model: AutoEye,
    optimizer: AdamW,
    scaler: GradScaler,
    trainloader: DataLoader,
    testloader: DataLoader,
) -> None:
    """Performs actual training"""

    finished_epochs = 0
    for epoch in range(1, Config.cfg.hyper.epochs + 1):
        epoch_loss: float = 0.0
        epoch_accuracy: float = 0.0
        start: float = time.time()

        # tqdm bar
        with tqdm(iter(trainloader)) as tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
            for batch_sample in tepoch:
                # enable mixed precision
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=Config.cfg.hyper.use_amp,
                ):
                    # data, targets
                    x, targets, _ = batch_sample
                    # targets = targets.unsqueeze(1)

                    # forward
                    logits = model(x)

                    # compute the loss
                    loss: torch.Tensor = F.binary_cross_entropy_with_logits(
                        logits, targets
                    )
                    epoch_loss += loss.item()

                # backprop and optimize
                if Config.cfg.hyper.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)
                # evaluate the accuracy
                epoch_accuracy += evaluate_accuracy(logits, targets)

        finished_epochs += 1

        # save model
        checkpoint = {
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs": Config.get_trained_epochs() + finished_epochs,
        }

        # check if models directory does not exist
        if not os.path.exists("models"):
            # create it if it does not exist
            os.mkdir("models")

        # save checkpoint
        torch.save(checkpoint, Config.model_path)
        if epoch % 10 == 0:
            torch.save(checkpoint, Config.model_path)

        # monitor loss and accuracy
        losses = epoch_loss / len(trainloader)
        accuracy = epoch_accuracy / len(trainloader)
        logging.info(f"Train Loss: {losses}, Train Accuracy: {accuracy:.2f}%")
        # evaluation
        if epoch % Config.cfg.hyper.eval_iters == 0:
            evaluate(model, testloader)
        # compute elapsed time of the epoch
        end: float = time.time()
        seconds_elapsed: float = end - start
        logging.info(
            f"Time elapsed: {int((seconds_elapsed)//60)} min {math.ceil(seconds_elapsed % 60)} s"
        )


if __name__ == "__main__":
    # torch.manual_seed(42)
    # torch.multiprocessing.set_start_method("spawn")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    train()
