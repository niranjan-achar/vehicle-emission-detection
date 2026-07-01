"""
Vehicle Rear Detection using Improved Integral Projection Method.

Paper Section: 2.2 Vehicle rear detection
Reference equations: (3), (4), (5)

The method uses:
1. Horizontal Integral Projection (HIP) to locate grooves
2. Standard Deviation Filtering (STD) to distinguish real vehicle rear from interference
3. Selects the groove with minimum standard deviation as vehicle rear position
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class RearDetector:
    """Detect vehicle rear position using integral projection."""

    def __init__(self, eps: int = 10, std_window: int = 5):
        """
        Initialize rear detector.

        Args:
            eps: Height of vehicle rear region (in pixels).
            std_window: Window size for standard deviation filtering.
        """
        self.eps = eps
        self.std_window = std_window

    def detect_rear(self, vehicle_mask: np.ndarray) -> int:
        """
        Detect vehicle rear position from vehicle mask.

        Args:
            vehicle_mask: Binary mask of vehicle region (H, W) where 255=vehicle.

        Returns:
            Y-coordinate of detected vehicle rear position.
        """
        if vehicle_mask.size == 0:
            return -1

        # Step 1: Calculate horizontal integral projection (HIP)
        hip = self._calc_horizontal_ip(vehicle_mask)

        if hip is None or len(hip) == 0:
            return -1

        # Step 2: Calculate standard deviation filtering curve
        std_curve = self._calc_std_filtering(hip)

        if std_curve is None or len(std_curve) == 0:
            return -1

        # Step 3: Extract first five groove locations from left half of HIP
        h = vehicle_mask.shape[0]
        grooves = self._find_grooves(hip, num_grooves=5, start_y=h // 2)

        if not grooves:
            return -1

        # Step 4: Select groove with minimum standard deviation
        rear_y = self._select_rear_by_std(grooves, std_curve)

        logger.debug(f"Detected vehicle rear at y={rear_y}, from grooves={grooves}")
        return rear_y

    @staticmethod
    def _calc_horizontal_ip(vehicle_mask: np.ndarray) -> np.ndarray:
        """
        Calculate horizontal integral projection.

        Paper Equation (3):
        HIP(y) = (1/(q2-q1)) * Σ(q=q1 to q2) I_obj(q, y)

        Args:
            vehicle_mask: Binary vehicle mask (H, W).

        Returns:
            HIP curve of length W.
        """
        if vehicle_mask.size == 0:
            return None

        # Find horizontal bounds of vehicle
        rows = np.any(vehicle_mask, axis=1)
        if not np.any(rows):
            return None

        row_indices = np.where(rows)[0]
        q1, q2 = row_indices[0], row_indices[-1] + 1

        if q2 - q1 <= 0:
            return None

        # Calculate HIP for each column
        h, w = vehicle_mask.shape
        hip = np.zeros(w)

        for y in range(w):
            hip[y] = np.mean(vehicle_mask[q1:q2, y].astype(np.float32))

        return hip

    @staticmethod
    def _calc_std_filtering(hip: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Calculate standard deviation filtering curve.

        Paper Equation (5):
        STD(y) = stdfilt(HIP(y))

        Args:
            hip: Horizontal integral projection curve.
            window: Window size for std filtering.

        Returns:
            Standard deviation curve.
        """
        if hip is None or len(hip) == 0:
            return None

        std_curve = np.zeros_like(hip)

        for i in range(len(hip)):
            start = max(0, i - window // 2)
            end = min(len(hip), i + window // 2 + 1)
            std_curve[i] = np.std(hip[start:end])

        return std_curve

    @staticmethod
    def _find_grooves(
        hip: np.ndarray,
        num_grooves: int = 5,
        start_y: int = 0,
    ) -> list[int]:
        """
        Find groove locations (local minima) in HIP curve.

        Paper Equation (4):
        y_rear = arg min_{y in [h/2, h]} Σ(i=y-eps to y) HIP(i)

        Args:
            hip: Horizontal integral projection curve.
            num_grooves: Number of grooves to extract.
            start_y: Start searching from this y position.

        Returns:
            List of y-coordinates of grooves.
        """
        if hip is None or len(hip) == 0:
            return []

        grooves = []
        h = len(hip)

        # Search from start_y to end
        search_range = hip[start_y:h]

        if len(search_range) == 0:
            return []

        # Find local minima
        for _ in range(num_grooves):
            if len(search_range) == 0:
                break

            min_idx = np.argmin(search_range)
            groove_y = start_y + min_idx

            grooves.append(groove_y)

            # Remove region around this groove for next search
            margin = 10
            start = max(0, min_idx - margin)
            end = min(len(search_range), min_idx + margin + 1)
            search_range = np.concatenate([search_range[:start], search_range[end:]])
            start_y += start

        return grooves

    @staticmethod
    def _select_rear_by_std(grooves: list[int], std_curve: np.ndarray) -> int:
        """
        Select the groove with minimum standard deviation as vehicle rear.

        Args:
            grooves: List of groove y-coordinates.
            std_curve: Standard deviation filtering curve.

        Returns:
            Y-coordinate of selected rear position.
        """
        if not grooves:
            return -1

        min_std = float("inf")
        rear_y = grooves[0]

        for groove_y in grooves:
            if 0 <= groove_y < len(std_curve):
                if std_curve[groove_y] < min_std:
                    min_std = std_curve[groove_y]
                    rear_y = groove_y

        return rear_y
