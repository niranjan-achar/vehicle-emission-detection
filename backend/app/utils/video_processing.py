from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List

import cv2

def process_video(
    input_path: Path,
    output_path: Path,
    paper_service,
    confidence_threshold: float,
    window_size: int = 100,
    alpha1: int = 10,
) -> tuple[int, List[float], int, float]:
    """
    Process video frames using paper-based detection service.

    Args:
        input_path: Input video file path.
        output_path: Output video file path.
        paper_service: Paper-based detection service instance.
        confidence_threshold: Confidence threshold for detections (unused for paper service).

    Returns:
        Tuple of (detections_count, timestamps, frame_count, duration_seconds).
    """

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
    smoke_window = deque(maxlen=max(1, window_size))
    alert_active = False

    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            break

        # Use paper-based detection
        detection_result = paper_service.detect_frame(frame)
        detections = detection_result.get("detections", [])
        annotated_frame = detection_result.get("frame_annotated", frame)

        smoke_frame = (
            1 if any(det.get("class_name") == "smoky" for det in detections) else 0
        )
        smoke_window.append(smoke_frame)

        window_triggered = (
            len(smoke_window) == smoke_window.maxlen and sum(smoke_window) >= alpha1
        )
        if window_triggered and not alert_active:
            detections_count += 1
            timestamps.append(round(frame_count / fps, 3))
            alert_active = True
        elif not window_triggered:
            alert_active = False

        if window_triggered:
            cv2.putText(
                annotated_frame,
                "SMOKE ALERT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        writer.write(annotated_frame)

        frame_count += 1

    capture.release()
    writer.release()

    duration_seconds = round(frame_count / fps, 3) if fps > 0 else 0.0
    return detections_count, timestamps, frame_count, duration_seconds
