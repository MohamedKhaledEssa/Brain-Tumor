"""Image preprocessing utilities.

Converts uploaded bytes into normalized tensors suitable for a
torchvision-style CNN (ImageNet stats, 224x224 RGB).
"""

from __future__ import annotations

import io

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

INPUT_SIZE: tuple[int, int] = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


def load_image(data: bytes) -> Image.Image:
    """Decode raw bytes to a PIL RGB image."""
    if not data:
        raise InvalidImageError("Empty file.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decode so we catch errors here
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Unsupported or corrupt image: {exc}") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Apply the standard preprocessing pipeline. Returns shape (3, H, W)."""
    return _transform(image)


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Convenience: decode bytes and produce a model-ready tensor."""
    return image_to_tensor(load_image(data))
