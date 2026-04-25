"""Training script for the brain tumor classifier.

Trains a torchvision CNN on a directory laid out as

    <data_root>/Training/<class_name>/*.{jpg,png,...}
    <data_root>/Testing/<class_name>/*.{jpg,png,...}

This matches the standard "Brain Tumor MRI Dataset" available on Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Run from the ``backend`` directory:

    python -m scripts.train --data-root /path/to/dataset --epochs 10

The trained weights are written to ``backend/models/brain_tumor.pth`` by
default; the API auto-loads them on next start.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.ml.classes import CLASS_NAMES
from app.ml.model import _build_backbone
from app.ml.preprocess import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")


def build_loaders(
    data_root: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(INPUT_SIZE[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dir = data_root / "Training"
    test_dir = data_root / "Testing"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {train_dir} and {test_dir} to exist with class subfolders."
        )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    if tuple(train_ds.classes) != CLASS_NAMES:
        logger.warning(
            "Dataset classes %s do not exactly match expected %s. "
            "Make sure folder names are: %s",
            train_ds.classes,
            CLASS_NAMES,
            ", ".join(CLASS_NAMES),
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, train_ds.classes


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    logger.info("Using device: %s", device)

    train_loader, test_loader, classes = build_loaders(
        Path(args.data_root), args.batch_size, args.num_workers
    )
    logger.info("Found classes: %s", classes)

    model = _build_backbone(args.arch, len(classes), pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "epoch %d batch %d/%d loss=%.4f",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    running_loss / batch_idx,
                )

        acc = evaluate(model, test_loader, device)
        logger.info("epoch %d test_acc=%.4f", epoch, acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_path)
            logger.info("Saved improved model to %s (acc=%.4f)", output_path, acc)

    logger.info("Best test accuracy: %.4f. Weights at: %s", best_acc, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the brain tumor classifier.")
    parser.add_argument("--data-root", required=True, type=str, help="Path to dataset root.")
    parser.add_argument("--arch", default="resnet18", help="resnet18 | resnet50 | mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained backbone.")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "models" / "brain_tumor.pth"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
