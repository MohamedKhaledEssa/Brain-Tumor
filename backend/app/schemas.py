"""Pydantic schemas for request/response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str = Field(..., description="Class label.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability.")


class Prediction(BaseModel):
    filename: str = Field(..., description="Original uploaded filename.")
    predicted_label: str = Field(..., description="Top-1 predicted class.")
    predicted_index: int = Field(..., ge=0, description="Index of the predicted class.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 probability.")
    probabilities: list[ClassProbability] = Field(
        ..., description="Per-class probabilities (sums to ~1.0)."
    )
    inference_ms: float = Field(..., ge=0.0, description="Inference time in milliseconds.")


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    count: int = Field(..., ge=0)


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    app_name: str
    version: str
    environment: str


class ModelInfo(BaseModel):
    architecture: str
    num_classes: int
    classes: list[str]
    weights_loaded: bool = Field(
        ..., description="True if trained weights were loaded; False if running in demo mode."
    )
    weights_path: str | None
    device: str
    input_size: tuple[int, int] = Field(default=(224, 224))


class ApiInfo(BaseModel):
    name: str
    version: str
    description: str
    docs_url: str = "/docs"
    health_url: str = "/health"


class ErrorResponse(BaseModel):
    detail: str
