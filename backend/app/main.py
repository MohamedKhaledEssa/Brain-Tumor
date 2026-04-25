"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.ml.model import TumorDetector
from app.routers import health, model_info, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    detector = TumorDetector(settings)
    app.state.detector = detector
    logger.info(
        "Model ready: arch=%s device=%s weights_loaded=%s",
        settings.model_architecture,
        detector.device,
        detector.weights_loaded,
    )
    try:
        yield
    finally:
        logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Backend API for the Brain Tumor Detection mobile app. "
            "Provides MRI image classification across four classes: "
            "glioma, meningioma, notumor, pituitary."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(model_info.router)
    app.include_router(predict.router)
    return app


app = create_app()
