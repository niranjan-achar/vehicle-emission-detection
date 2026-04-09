from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

  
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(description="[x1, y1, x2, y2]")


class ImageDetectionResponse(BaseModel):
    file_name: str
    detections: List[Detection]
    detections_count: int
    processed_image_base64: str
    processed_image_path: str


class VideoDetectionResponse(BaseModel):
    file_name: str
    detections_count: int
    timestamps: List[float]
    processed_video_path: str
    total_frames: int
    duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    environment: str
    smoke_class_names: List[str] = []
    resolved_smoke_class_ids: List[int] = []
    model_classes: List[str] = []
    warning: str | None = None


class DashboardSummaryResponse(BaseModel):
    total_uploads: int
    total_polluting_detections: int
    image_uploads: int
    video_uploads: int
    last_updated: datetime | None = None
