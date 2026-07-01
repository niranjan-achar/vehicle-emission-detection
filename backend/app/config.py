from functools import lru_cache
from pathlib import Path
from typing import List

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
    svm_model_path: Path = BASE_DIR / "weights" / "paper_svm_models.pkl"
    uploads_dir: Path = BASE_DIR / "storage" / "uploads"
    processed_dir: Path = BASE_DIR / "storage" / "processed"
    json_db_path: Path = BASE_DIR / "storage" / "detections.json"

    bg_samples: int = 20
    bg_min_dist: int = 20
    bg_req_matches: int = 2
    eta_match: float = 0.15
    s_min: int = 1500
    s_max: int = 50000
    delta_min: float = 0.3
    delta_max: float = 1.5
    eps: int = 10
    std_window: int = 5
    key_region_width_ratio: float = 0.7
    key_region_height: int = 10
    key_region_margin: int = 5
    video_window_size: int = 100
    alpha1: int = 10

    default_confidence_threshold: float = 0.20
    max_upload_size_mb: int = 100
    smoke_class_names: str = "smoke"

    allowed_image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    mongo_uri: str | None = None
    mongo_db_name: str = "vehicle_emission"
    mongo_collection_name: str = "detections"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def smoke_class_names_list(self) -> List[str]:
        return [
            item.strip().lower()
            for item in self.smoke_class_names.split(",")
            if item.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
