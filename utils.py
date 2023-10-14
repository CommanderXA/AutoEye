import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from autoeye.config import Config

from autoeye.model import AutoEye


@torch.no_grad
def evaluate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Evaluate accuracy of a particular batch"""
    diff = torch.abs(F.sigmoid(logits) - targets)
    # Set a threshold value (e.g., 0.5)
    threshold = 0.5
    # Convert sigmoid outputs to 0 or 1 based on the threshold
    binary_outputs = torch.where(diff >= threshold, torch.tensor(1), torch.tensor(0))
    correct = torch.sum(binary_outputs, dim=0).item()
    sample = targets.size(0)
    accuracy = 100 * (correct / sample)
    return accuracy


@torch.no_grad
def evaluate(model: AutoEye, dataloader: DataLoader):
    model.eval()
    epoch_accuracy = 0.0
    epoch_loss = 0
    with tqdm(iter(dataloader)) as tepoch:
        tepoch.set_description("Evaluating")
        for batch_sample in tepoch:
            # enable mixed precision
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=Config.cfg.hyper.use_amp,
            ):
                # data, targets
                x, targets, _ = batch_sample

                # forward
                logits = model(x)

                # compute the loss
                loss: torch.Tensor = F.binary_cross_entropy_with_logits(logits, targets)
                epoch_accuracy += evaluate_accuracy(logits, targets)
                epoch_loss += loss.item()

    accuracy = epoch_accuracy / len(dataloader)
    loss = epoch_loss / len(dataloader)
    print(f"\n|\n| Accuracy: {accuracy:.2f}%")
    print(f"| Loss: {loss}\n|\n")
    model.train()

@torch.no_grad
def evaluate_test(model: AutoEye, dataloader: DataLoader):
    model.eval()
    epoch_accuracy = 0.0
    epoch_loss = 0
    with tqdm(iter(dataloader)) as tepoch:
        tepoch.set_description("Evaluating")
        for batch_sample in tepoch:
            # enable mixed precision
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=Config.cfg.hyper.use_amp,
            ):
                # data, targets
                x, targets, _ = batch_sample

                # forward
                logits = model(x)

                # compute the loss
                loss: torch.Tensor = F.binary_cross_entropy_with_logits(logits, targets)
                epoch_accuracy += evaluate_accuracy(logits, targets)
                epoch_loss += loss.item()

    accuracy = epoch_accuracy / len(dataloader)
    loss = epoch_loss / len(dataloader)
    print(f"\n|\n| Accuracy: {accuracy:.2f}%")
    print(f"| Loss: {loss}\n|\n")
    model.train()