import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from autoeye.config import Config
from autoeye.model import AutoEye


@torch.no_grad
def evaluate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Evaluate accuracy of a particular batch"""
    # Convert sigmoid outputs to 0 or 1 based on the threshold
    threshold = 0.5
    binary_outputs = torch.where(
        F.sigmoid(logits) >= threshold, torch.tensor(1), torch.tensor(0)
    )
    correct = (binary_outputs == targets).sum().item()
    sample = targets.size(0)
    accuracy = 100 * (correct / sample)
    return accuracy


@torch.no_grad
def evaluate_accuracy_multiclass(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Evaluate accuracy of a particular batch"""
    # Convert sigmoid outputs to 0 or 1 based on the threshold
    logits = F.softmax(logits, 1)
    logits = torch.argmax(logits, 1)
    targets = torch.argmax(targets, 1)
    correct = (logits == targets).sum().item()
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
                x, targets = batch_sample

                # forward
                logits = model(x)

                # compute the loss
                loss: torch.Tensor = F.cross_entropy(logits, targets)
                epoch_accuracy += evaluate_accuracy_multiclass(logits, targets)
                epoch_loss += loss.item()

    accuracy = epoch_accuracy / len(dataloader)
    loss = epoch_loss / len(dataloader)
    print(f"\n|\n| Accuracy: {accuracy:.2f}%")
    print(f"| Loss: {loss}\n|\n")
    model.train()
    return accuracy
