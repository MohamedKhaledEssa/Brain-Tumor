"""Tumor detection model wrapper.

Loads a torchvision CNN with a custom 4-class head. If a weights file
exists at ``settings.model_path``, it is loaded; otherwise the model
runs in **demo mode** with a randomly initialized classification head.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from app.config import Settings
from app.ml import classes as class_mod
from app.ml.preprocess import INPUT_SIZE, image_to_tensor

logger = logging.getLogger(__name__)


def _resolve_device(preferred: str) -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def _build_backbone(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    arch = arch.lower()
    weights_arg = None
    if arch == "resnet18":
        if pretrained:
            weights_arg = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights_arg)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if arch == "resnet50":
        if pretrained:
            weights_arg = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=weights_arg)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if arch == "mobilenet_v3_small":
        if pretrained:
            weights_arg = models.MobileNet_V3_Small_Weights.DEFAULT
        net = models.mobilenet_v3_small(weights=weights_arg)
        in_features = cast(nn.Linear, net.classifier[-1]).in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
        return net
    raise ValueError(f"Unsupported model architecture: {arch}")


class TumorDetector:
    """Stateful inference wrapper around a torch CNN."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.classes: tuple[str, ...] = class_mod.CLASS_NAMES
        self.device = _resolve_device(settings.device)
        self.input_size: tuple[int, int] = INPUT_SIZE

        self.model = _build_backbone(
            settings.model_architecture,
            len(self.classes),
            pretrained=settings.use_pretrained_backbone,
        )
        self.weights_loaded = False
        self.weights_path: Path | None = None

        self._maybe_load_weights(settings.model_path)

        self.model.to(self.device)
        self.model.eval()

    # ----- weights -----
    def _maybe_load_weights(self, path: Path) -> None:
        if not path.exists():
            logger.warning(
                "No model weights found at %s — running in DEMO mode "
                "(predictions are not clinically meaningful).",
                path,
            )
            return
        try:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            self.weights_loaded = True
            self.weights_path = path
            logger.info("Loaded model weights from %s", path)
        except (RuntimeError, FileNotFoundError, KeyError) as exc:
            logger.error("Failed to load weights from %s: %s", path, exc)

    # ----- inference -----
    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[int, list[float], float]:
        """Return (predicted_index, probabilities, inference_ms)."""
        tensor = image_to_tensor(image).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        inference_ms = (time.perf_counter() - start) * 1000.0
        predicted_index = int(max(range(len(probs)), key=probs.__getitem__))
        return predicted_index, [float(p) for p in probs], inference_ms

    @torch.inference_mode()
    def predict_batch(
        self, images: list[Image.Image]
    ) -> list[tuple[int, list[float], float]]:
        """Vectorized batched inference."""
        if not images:
            return []
        tensors = torch.stack([image_to_tensor(im) for im in images]).to(self.device)
        start = time.perf_counter()
        logits = self.model(tensors)
        probs = torch.softmax(logits, dim=1).cpu().tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_item_ms = elapsed_ms / len(images)
        results: list[tuple[int, list[float], float]] = []
        for row in probs:
            idx = int(max(range(len(row)), key=row.__getitem__))
            results.append((idx, [float(p) for p in row], per_item_ms))
        return results
