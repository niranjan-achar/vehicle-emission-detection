from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from app.models.schema import Detection
from ultralytics import YOLO

logger = logging.getLogger(__name__)

  
@dataclass
class InferenceOutput:
    detections: List[Detection]
    annotated_frame: np.ndarray


class YoloService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model: YOLO | None = None
        self.model_loaded: bool = False
        self.model_warning: str | None = None

    def load(self, fallback_model_name: str | None = None) -> None:
        try:
            if not self.model_path.exists() or self.model_path.stat().st_size == 0:
                raise FileNotFoundError(f"Model file not found or empty: {self.model_path}")

            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            self.model_warning = None
            logger.info("YOLO model loaded from %s", self.model_path)
            return
        except Exception as primary_error:
            if not fallback_model_name:
                raise

            logger.warning("Primary model load failed: %s", primary_error)
            self.model = YOLO(fallback_model_name)
            self.model_path = Path(fallback_model_name)
            self.model_loaded = True
            self.model_warning = (
                "Custom smoke model failed to load. Running with fallback model "
                f"{fallback_model_name}. Replace backend/weights/best.pt for smoke-specific detection."
            )
            logger.warning(self.model_warning)

    def predict_image(self, image: np.ndarray, conf: float, class_ids: List[int] | None = None) -> InferenceOutput:
        self._ensure_ready()
        assert self.model is not None

        results = self.model.predict(source=image, conf=conf, classes=class_ids, verbose=False)
        result = results[0]
        detections = self._extract_detections(result)
        annotated = result.plot()
        return InferenceOutput(detections=detections, annotated_frame=annotated)

    def predict_frame(self, frame: np.ndarray, conf: float, class_ids: List[int] | None = None) -> InferenceOutput:
        return self.predict_image(frame, conf, class_ids=class_ids)

    def get_available_classes(self) -> List[str]:
        self._ensure_ready()
        assert self.model is not None
        names_obj = self.model.names

        if isinstance(names_obj, dict):
            return [str(names_obj[key]) for key in sorted(names_obj.keys())]

        if isinstance(names_obj, list):
            return [str(item) for item in names_obj]

        return []

    def resolve_class_ids(self, target_names: List[str]) -> List[int]:
        classes = [name.lower() for name in self.get_available_classes()]
        target_set = {name.lower() for name in target_names}
        resolved = [idx for idx, name in enumerate(classes) if name in target_set]
        return resolved

    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        success, encoded_img = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode processed image")
        return base64.b64encode(encoded_img.tobytes()).decode("utf-8")

    @staticmethod
    def encode_image_to_jpg_bytes(image: np.ndarray) -> bytes:
        success, encoded_img = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode processed image")
        return encoded_img.tobytes()

    def _ensure_ready(self) -> None:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

    def _extract_detections(self, result) -> List[Detection]:
        detections: List[Detection] = []
        names = result.names if hasattr(result, "names") else {}

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_name = str(names.get(class_id, class_id))

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                )
            )

        return detections
