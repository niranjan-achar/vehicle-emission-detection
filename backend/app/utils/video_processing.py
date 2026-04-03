from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
from app.services.yolo_service import YoloService


def process_video(
    input_path: Path,
    output_path: Path,
    yolo_service: YoloService,
    confidence_threshold: float,
    smoke_class_ids: List[int] | None = None,
) -> tuple[int, List[float], int, float]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Unable to open video file")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    detections_count = 0
    timestamps: List[float] = []

    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            break

        inference = yolo_service.predict_frame(
            frame,
            conf=confidence_threshold,
            class_ids=smoke_class_ids,
        )
        writer.write(inference.annotated_frame)

        if inference.detections:
            detections_count += len(inference.detections)
            timestamps.append(round(frame_count / fps, 3))

        frame_count += 1

    capture.release()
    writer.release()

    duration_seconds = round(frame_count / fps, 3) if fps > 0 else 0.0
    return detections_count, timestamps, frame_count, duration_seconds
