"""Endpoints exposing model metadata."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.ml.classes import CLASS_DESCRIPTIONS, CLASS_NAMES
from app.ml.model import TumorDetector
from app.schemas import ModelInfo

router = APIRouter(tags=["model"])


def _get_detector(request: Request) -> TumorDetector:
    return request.app.state.detector


@router.get("/classes", summary="List supported tumor classes")
def list_classes() -> dict[str, object]:
    return {
        "classes": list(CLASS_NAMES),
        "descriptions": CLASS_DESCRIPTIONS,
        "count": len(CLASS_NAMES),
    }


@router.get("/model/info", response_model=ModelInfo, summary="Model metadata")
def model_info(request: Request) -> ModelInfo:
    detector = _get_detector(request)
    return ModelInfo(
        architecture=detector.settings.model_architecture,
        num_classes=len(detector.classes),
        classes=list(detector.classes),
        weights_loaded=detector.weights_loaded,
        weights_path=str(detector.weights_path) if detector.weights_path else None,
        device=str(detector.device),
        input_size=detector.input_size,
    )
