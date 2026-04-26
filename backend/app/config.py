"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime configuration."""

    app_name: str = os.getenv("APP_NAME", "Brain Tumor Detection API")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    environment: str = os.getenv("ENVIRONMENT", "development")

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)

    model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "brain_tumor.pth"))
        )
    )
    model_architecture: str = os.getenv("MODEL_ARCH", "resnet18")
    device: str = os.getenv("DEVICE", "auto")  # "auto" | "cpu" | "cuda"

    max_upload_bytes: int = _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    max_batch_size: int = _env_int("MAX_BATCH_SIZE", 16)

    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("CORS_ORIGINS", ["*"])
    )

    use_pretrained_backbone: bool = _env_bool("USE_PRETRAINED_BACKBONE", False)


def get_settings() -> Settings:
    return Settings()
