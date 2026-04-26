"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile

from app.config import Settings, get_settings
from app.errors import http_400, http_413, http_415
from app.ml.model import TumorDetector
from app.ml.preprocess import InvalidImageError, load_image
from app.schemas import BatchPrediction, ClassProbability, Prediction

router = APIRouter(tags=["predict"])

_ALLOWED_PREFIXES = ("image/",)


def _get_detector(request: Request) -> TumorDetector:
    return request.app.state.detector


async def _read_image(upload: UploadFile, settings: Settings) -> tuple[bytes, str]:
    if upload.content_type and not upload.content_type.startswith(_ALLOWED_PREFIXES):
        raise http_415(
            f"Unsupported content type '{upload.content_type}'. Expected an image/* upload."
        )
    data = await upload.read()
    if len(data) == 0:
        raise http_400("Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise http_413(
            f"File '{upload.filename}' exceeds the maximum size of "
            f"{settings.max_upload_bytes} bytes."
        )
    return data, upload.filename or "upload"


def _to_prediction(
    filename: str,
    detector: TumorDetector,
    predicted_index: int,
    probs: list[float],
    inference_ms: float,
) -> Prediction:
    classes = detector.classes
    return Prediction(
        filename=filename,
        predicted_label=classes[predicted_index],
        predicted_index=predicted_index,
        confidence=probs[predicted_index],
        probabilities=[
            ClassProbability(label=label, probability=p)
            for label, p in zip(classes, probs, strict=True)
        ],
        inference_ms=inference_ms,
    )


@router.post(
    "/predict",
    response_model=Prediction,
    summary="Classify a single MRI image",
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="MRI scan image (JPEG, PNG, etc.)"),
) -> Prediction:
    settings = get_settings()
    detector = _get_detector(request)
    data, filename = await _read_image(file, settings)
    try:
        image = load_image(data)
    except InvalidImageError as exc:
        raise http_400(str(exc)) from exc
    idx, probs, ms = detector.predict(image)
    return _to_prediction(filename, detector, idx, probs, ms)


@router.post(
    "/predict/batch",
    response_model=BatchPrediction,
    summary="Classify a batch of MRI images",
)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more MRI images."),
) -> BatchPrediction:
    settings = get_settings()
    detector = _get_detector(request)
    if not files:
        raise http_400("No files provided.")
    if len(files) > settings.max_batch_size:
        raise http_400(
            f"Batch size {len(files)} exceeds the configured maximum "
            f"of {settings.max_batch_size}."
        )

    images = []
    filenames: list[str] = []
    for upload in files:
        data, filename = await _read_image(upload, settings)
        try:
            images.append(load_image(data))
        except InvalidImageError as exc:
            raise http_400(f"{filename}: {exc}") from exc
        filenames.append(filename)

    results = detector.predict_batch(images)
    predictions = [
        _to_prediction(filenames[i], detector, idx, probs, ms)
        for i, (idx, probs, ms) in enumerate(results)
    ]
    return BatchPrediction(predictions=predictions, count=len(predictions))
