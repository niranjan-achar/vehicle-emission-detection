from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from app.config import get_settings
from app.models.schema import (
    DashboardSummaryResponse,
    Detection,
    ImageDetectionResponse,
    VideoDetectionResponse,
)
from app.utils.video_processing import process_video
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detect", tags=["Detection"])


def _validate_upload_extension(file_name: str, valid_extensions: list[str]) -> None:
    suffix = Path(file_name).suffix.lower()
    if suffix not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {suffix}",
        )


def _ensure_model_loaded(request: Request) -> None:
    if not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health endpoint for details.",
        )


def _to_detection_payload(det: dict) -> Detection:
    bbox = det.get("bbox", (0.0, 0.0, 0.0, 0.0))
    if len(bbox) == 4:
        x1, y1, width, height = bbox
        bbox_xyxy = [float(x1), float(y1), float(x1 + width), float(y1 + height)]
    else:
        bbox_xyxy = [0.0, 0.0, 0.0, 0.0]

    return Detection(
        class_id=int(det.get("class_id", 0)),
        class_name=str(det.get("class_name", "non_smoky")),
        confidence=float(det.get("confidence", 0.0)),
        bbox=bbox_xyxy,
    )


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = Query(default=None, ge=0.0, le=1.0),
) -> ImageDetectionResponse:
    settings = get_settings()
    _ensure_model_loaded(request)
    _validate_upload_extension(file.filename or "", settings.allowed_image_extensions)

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="Image file is too large")

    np_buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Use paper-based detection
    detection_result = request.app.state.paper_service.detect_frame(image)
    detections = detection_result.get("detections", [])
    annotated_frame = detection_result.get("frame_annotated", image)

    output_name = f"{uuid.uuid4().hex}.jpg"
    output_path = settings.processed_dir / output_name
    _, encoded_jpg = cv2.imencode(".jpg", annotated_frame)
    output_path.write_bytes(encoded_jpg.tobytes())

    response = ImageDetectionResponse(
        file_name=file.filename or output_name,
        detections=[_to_detection_payload(d) for d in detections],
        detections_count=len(detections),
        processed_image_base64=base64.b64encode(encoded_jpg.tobytes()).decode("utf-8"),
        processed_image_path=f"/static/processed/{output_name}",
    )

    record = {
        "media_type": "image",
        "file_name": file.filename,
        "detections_count": len(detections),
        "confidence_threshold": confidence or settings.default_confidence_threshold,
    }

    request.app.state.storage_service.save_detection_record(record)
    # trigger notifications (async, non-blocking)
    try:
        request.app.state.notification_service.notify_async(record)
    except Exception:
        logger.exception("Failed to schedule notification for image")

    return response


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = Query(default=None, ge=0.0, le=1.0),
) -> VideoDetectionResponse:
    settings = get_settings()
    _ensure_model_loaded(request)
    _validate_upload_extension(file.filename or "", settings.allowed_video_extensions)

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="Video file is too large")

    input_name = f"{uuid.uuid4().hex}{Path(file.filename or '.mp4').suffix}"
    input_path = settings.uploads_dir / input_name
    input_path.write_bytes(content)

    output_name = f"processed-{uuid.uuid4().hex}.mp4"
    output_path = settings.processed_dir / output_name

    threshold = confidence if confidence is not None else settings.default_confidence_threshold

    try:
        detections_count, timestamps, total_frames, duration = process_video(
            input_path=input_path,
            output_path=output_path,
            paper_service=request.app.state.paper_service,
            confidence_threshold=threshold,
            window_size=settings.video_window_size,
            alpha1=settings.alpha1,
        )
    except Exception as exc:
        logger.exception("Video processing failed")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}") from exc
    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)

    record = {
        "media_type": "video",
        "file_name": file.filename,
        "detections_count": detections_count,
        "confidence_threshold": threshold,
    }

    request.app.state.storage_service.save_detection_record(record)
    try:
        request.app.state.notification_service.notify_async(record)
    except Exception:
        logger.exception("Failed to schedule notification for video")

    return VideoDetectionResponse(
        file_name=file.filename or output_name,
        detections_count=detections_count,
        timestamps=timestamps,
        processed_video_path=f"/static/processed/{output_name}",
        total_frames=total_frames,
        duration_seconds=duration,
    )


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_summary(request: Request) -> DashboardSummaryResponse:
    summary = request.app.state.storage_service.get_summary()
    return DashboardSummaryResponse(**summary)
