import logging
import math
import os
import time

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig

from tqdm import tqdm
import pandas as pd

from autoeye.config import Config
from autoeye.dataset import AutoDataset
from autoeye.model import AutoEye


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    Config.setup(cfg, log, train=False)

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
    # model = torch.compile(model)
    model.eval()

    if cfg.hyper.pretrained and os.path.exists(f"{Config.model_path[:-3]}_best.pt"):
        checkpoint = torch.load(f"{Config.model_path[:-3]}_best.pt")
        model.load(checkpoint)
        Config.set_trained_epochs(checkpoint["epochs"])

    logging.info(
        f"Model parameters amount: {model.get_parameters_amount():,} (Trained on {Config.get_trained_epochs()} epochs)"
    )

    # test dataset
    testdataset = AutoDataset(cfg.data.csv_files[1])
    # test dataloader
    testloader = DataLoader(dataset=testdataset, batch_size=1, shuffle=False)
    predict(model, testloader)


def predict(
    model: AutoEye,
    testloader: DataLoader,
) -> None:
    """Performs prediction"""

    start: float = time.time()
    predictions = {"file_index": [], "class": []}

    # tqdm bar
    with tqdm(iter(testloader)) as tepoch:
        tepoch.set_description(f"Predicting")
        for batch_sample in tepoch:
            # enable mixed precision
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=Config.cfg.hyper.use_amp,
            ):
                # data, targets
                index, x = batch_sample

                # forward
                logits = model(x)

                logits = torch.argmax(logits, 1).item()
                if logits > 0:
                    logits = 1

                predictions["file_index"].append(index.item())
                predictions["class"].append(logits)

    # compute elapsed time of the epoch
    end: float = time.time()
    seconds_elapsed: float = end - start
    logging.info(
        f"Time elapsed: {int((seconds_elapsed)//60)} min {math.ceil(seconds_elapsed % 60)} s"
    )

    df = pd.DataFrame(predictions)
    df.to_csv("./models/test.csv", index=False)


if __name__ == "__main__":
    # torch.manual_seed(42)
    # torch.multiprocessing.set_start_method("spawn")
    main()
