"""
Paper-Based Smoke Detection Service.

Orchestrates the complete pipeline:
1. Background subtraction (ViBe)
2. Vehicle candidate extraction and filtering
3. Vehicle rear detection (integral projection)
4. Key region extraction
5. Feature extraction (ART, GLCM, DWT)
6. SVM voting classification

This service replaces the YoloService while maintaining the same API interface.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from app.classifiers.voting import SVMVotingClassifier
from app.features.artificial import ArtificialFeatureExtractor
from app.features.dwt import DWTFeatureExtractor
from app.features.glcm import GLCMFeatureExtractor
from app.features.key_region import KeyRegionExtractor
from app.rear_detection.integral_projection import RearDetector
from app.vehicle_detection.background_subtraction import BackgroundSubtractor
from app.vehicle_detection.vehicle_filter import VehicleFilter

logger = logging.getLogger(__name__)


class PaperService:
    """Paper-based smoke detection service."""

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize paper-based detection service.

        Args:
            config: Configuration dictionary with parameters.
        """
        self.config = config or {}
        self.model_bundle_path = Path(
            self.config.get(
                'svm_model_path',
                Path(__file__).resolve().parents[2] / 'weights' / 'paper_svm_models.pkl',
            )
        )

        # Initialize components
        self.bg_subtractor = BackgroundSubtractor(
            samples=self.config.get('bg_samples', 20),
            min_dist=self.config.get('bg_min_dist', 20),
            req_matches=self.config.get('bg_req_matches', 2),
        )

        self.vehicle_filter = VehicleFilter(
            eta_match=self.config.get('eta_match', 0.15),
            s_min=self.config.get('s_min', 1500),
            s_max=self.config.get('s_max', 50000),
            delta_min=self.config.get('delta_min', 0.3),
            delta_max=self.config.get('delta_max', 1.5),
        )

        self.rear_detector = RearDetector(
            eps=self.config.get('eps', 10),
            std_window=self.config.get('std_window', 5),
        )

        self.key_region_extractor = KeyRegionExtractor(
            width_ratio=self.config.get('key_region_width_ratio', 0.7),
            height=self.config.get('key_region_height', 10),
            margin=self.config.get('key_region_margin', 5),
        )

        self.art_extractor = ArtificialFeatureExtractor()
        self.glcm_extractor = GLCMFeatureExtractor()
        self.dwt_extractor = DWTFeatureExtractor()

        # SVM voting classifier
        self.voting_classifier = SVMVotingClassifier()

        # SVM classifiers will be loaded later
        self.svms = {}
        self.scaler = None

        # Frame state for video processing
        self.prev_frame = None
        self.bg_frame = None

        logger.info("PaperService initialized")

    def load(self) -> bool:
        """
        Load pre-trained models and initialize service.

        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            if self.model_bundle_path.exists():
                with self.model_bundle_path.open('rb') as handle:
                    bundle = pickle.load(handle)

                if isinstance(bundle, dict) and 'classifier' in bundle:
                    self.voting_classifier = bundle['classifier']
                    logger.info("Loaded trained SVM bundle from %s", self.model_bundle_path)
                elif isinstance(bundle, SVMVotingClassifier):
                    self.voting_classifier = bundle
                    logger.info("Loaded trained SVM classifier from %s", self.model_bundle_path)
                else:
                    logger.warning(
                        "Unsupported SVM bundle format in %s; using heuristic fallback",
                        self.model_bundle_path,
                    )
            else:
                logger.warning(
                    "Trained SVM bundle not found at %s; using heuristic fallback",
                    self.model_bundle_path,
                )

            logger.info("PaperService loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading PaperService: {e}")
            return False

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Detect smoke-emitting vehicles in a single frame.

        API-compatible with YoloService.predict_frame.

        Args:
            frame: Input frame (H, W, 3) BGR.

        Returns:
            Detection result dictionary with:
            - 'detections': list of detection dicts with keys:
                - 'class_name': 'smoky' or 'non_smoky'
                - 'confidence': float in [0, 1]
                - 'bbox': (x, y, w, h)
                - 'features': dict of all extracted features
            - 'frame_annotated': annotated frame with bounding boxes
        """
        result = {
            'detections': [],
            'frame_annotated': frame.copy(),
        }

        if frame is None or frame.size == 0:
            return result

        try:
            # Step 1: Background subtraction
            foreground = self.bg_subtractor.apply(frame)

            if foreground is None or foreground.size == 0:
                return result

            # Step 2: Vehicle candidate extraction and filtering
            vehicles = self.vehicle_filter.filter_candidates(foreground, frame)

            if not vehicles:
                return result

            # Step 3: For each vehicle, detect rear and extract features
            annotated = frame.copy()

            for vehicle in vehicles:
                bbox = vehicle['bbox']
                x, y, w, h = bbox
                vehicle_mask = vehicle['mask']

                # Detect vehicle rear
                rear_y = self.rear_detector.detect_rear(vehicle_mask)

                if rear_y < 0:
                    continue

                # Extract key region
                key_region, metadata = self.key_region_extractor.extract_key_region(
                    frame, bbox, rear_y
                )

                if key_region is None:
                    continue

                # Extract features
                features = self._extract_all_features(key_region)

                # Predict class
                class_name, confidence = self._predict_class(features)

                detection = {
                    'class_id': 1 if class_name == 'smoky' else 0,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'features': features,
                    'rear_y': rear_y,
                }

                result['detections'].append(detection)

                # Annotate frame
                color = (0, 255, 0) if class_name == 'non_smoky' else (0, 0, 255)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    annotated,
                    f"{class_name} ({confidence:.2f})",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            result['frame_annotated'] = annotated

        except Exception as e:
            logger.error(f"Error in detect_frame: {e}")

        return result

    def detect_image(self, image_path: str) -> dict:
        """
        Detect smoke-emitting vehicles in an image file.

        API-compatible with YoloService.predict_image.

        Args:
            image_path: Path to image file.

        Returns:
            Detection result dictionary.
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return {'detections': [], 'frame_annotated': None}

            return self.detect_frame(frame)

        except Exception as e:
            logger.error(f"Error detecting image: {e}")
            return {'detections': [], 'frame_annotated': None}

    def _extract_all_features(self, key_region: np.ndarray) -> dict:
        """
        Extract all 36 features from key region.

        Returns:
            Dictionary with all feature values.
        """
        features = {}

        try:
            # ART features (4)
            art_features = self.art_extractor.extract_features(key_region)
            features.update(art_features)

            # GLCM features (20)
            glcm_features = self.glcm_extractor.extract_features(key_region)
            features.update(glcm_features)

            # DWT features (12)
            dwt_features = self.dwt_extractor.extract_features(key_region)
            features.update(dwt_features)

        except Exception as e:
            logger.error(f"Error extracting features: {e}")

        return features

    def _predict_class(self, features: dict) -> tuple[str, float]:
        """
        Predict class (smoky/non_smoky) using SVM voting.

        For now, returns a simple heuristic based on feature values.
        In production, would use trained SVM classifiers.

        Args:
            features: Dictionary of extracted features.

        Returns:
            Tuple of (class_name, confidence).
        """
        try:
            class_name, confidence, voting_details = self.voting_classifier.predict(features)
            logger.debug(f"SVM prediction: {class_name} ({confidence:.3f}), details: {voting_details}")
            return class_name, confidence
        except Exception as e:
            logger.error(f"Error predicting class: {e}")
            return 'non_smoky', 0.5
