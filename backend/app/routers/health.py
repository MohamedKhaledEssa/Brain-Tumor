"""Health and root endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import ApiInfo, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/", response_model=ApiInfo, summary="API information")
def root(settings: Settings = Depends(get_settings)) -> ApiInfo:
    return ApiInfo(
        name=settings.app_name,
        version=settings.app_version,
        description=(
            "FastAPI backend for the Brain Tumor Detection mobile app. "
            "Upload an MRI image to /predict to receive a class prediction."
        ),
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
