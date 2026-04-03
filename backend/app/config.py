from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "Vehicle Emission Detection API"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_path: Path = BASE_DIR / "weights" / "best.pt"
    fallback_model_name: str = "yolov8n.pt"
    allow_fallback_model: bool = True
    uploads_dir: Path = BASE_DIR / "storage" / "uploads"
    processed_dir: Path = BASE_DIR / "storage" / "processed"
    json_db_path: Path = BASE_DIR / "storage" / "detections.json"

    default_confidence_threshold: float = 0.20
    max_upload_size_mb: int = 100
    smoke_class_names: List[str] = ["smoke"]

    allowed_image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    mongo_uri: str | None = None
    mongo_db_name: str = "vehicle_emission"
    mongo_collection_name: str = "detections"

    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | List[str]) -> List[str]:
        if isinstance(value, list):
            return value
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("smoke_class_names", mode="before")
    @classmethod
    def parse_smoke_class_names(cls, value: str | List[str]) -> List[str]:
        if isinstance(value, list):
            return [item.strip().lower() for item in value if item.strip()]
        return [item.strip().lower() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
