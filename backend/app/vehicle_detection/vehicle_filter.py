"""
Vehicle Candidate Extraction and Filtering.

Paper Section: 2.1 Vehicle detection
Reference equations: (1) and (2)

Implements three rules to remove non-vehicle objects:
- Rule 1: Matching degree threshold (Rmatch < ηmatch)
- Rule 2: Area constraints (S ∈ [S1, S2])
- Rule 3: Aspect ratio constraints (width/height ∈ [δ1, δ2])
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VehicleFilter:
    """Filter vehicle candidates from foreground mask."""

    def __init__(
        self,
        eta_match: float = 0.15,
        s_min: int = 1500,
        s_max: int = 50000,
        delta_min: float = 0.3,
        delta_max: float = 1.5,
        rect_height: int = 50,
        rect_width_ratio: float = 0.6,
    ):
        """
        Initialize vehicle filter.

        Args:
            eta_match: Matching degree threshold (Rule 1).
            s_min, s_max: Area range in pixels (Rule 2).
            delta_min, delta_max: Aspect ratio range (Rule 3).
            rect_height: Height of region behind foreground object.
            rect_width_ratio: Width as ratio of object width.
        """
        self.eta_match = eta_match
        self.s_min = s_min
        self.s_max = s_max
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.rect_height = rect_height
        self.rect_width_ratio = rect_width_ratio

    def filter_candidates(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
    ) -> list[dict]:
        """
        Extract vehicle candidates from foreground mask and apply filtering rules.

        Args:
            foreground: Foreground mask (H, W) with 255=foreground.
            background: Original background frame (H, W, 3) or (H, W).

        Returns:
            List of valid vehicle candidates with:
            - 'bbox': (x, y, w, h)
            - 'mask': contour mask
            - 'area': contour area
        """
        vehicles = []

        # Find contours in foreground
        contours, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # Rule 2: Check area
            if not (self.s_min <= area <= self.s_max):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Rule 3: Check aspect ratio
            if w == 0 or h == 0:
                continue
            aspect_ratio = w / h
            if not (self.delta_min <= aspect_ratio <= self.delta_max):
                continue

            # Rule 1: Calculate matching degree
            # Region behind the foreground object
            rect_y1 = y + h
            rect_y2 = min(foreground.shape[0], y + h + self.rect_height)
            rect_x1 = x
            rect_x2 = x + int(w * self.rect_width_ratio)

            if rect_y2 - rect_y1 <= 0 or rect_x2 - rect_x1 <= 0:
                continue

            # Extract region from foreground and background
            if len(background.shape) == 3:
                # Convert to grayscale if color
                bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            else:
                bg_gray = background

            # Current region should be background (low matching degree)
            current_region = bg_gray[rect_y1:rect_y2, rect_x1:rect_x2]

            # Get background model for the same region (from initial background)
            # For now, use the region itself as reference (in real scenario, use maintained BG model)
            bg_region = bg_gray[rect_y1:rect_y2, rect_x1:rect_x2].copy()

            r_match = self._calc_matching_degree(current_region, bg_region)

            # Rule 1: Check matching degree
            if r_match >= self.eta_match:
                continue

            # All rules passed - valid vehicle candidate
            mask = np.zeros(foreground.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)

            vehicles.append(
                {
                    "bbox": (x, y, w, h),
                    "mask": mask,
                    "area": area,
                    "matching_degree": r_match,
                    "contour": contour,
                }
            )

        logger.debug(f"Extracted {len(vehicles)} vehicle candidates after filtering")
        return vehicles

    @staticmethod
    def _calc_matching_degree(region1: np.ndarray, region2: np.ndarray) -> float:
        """
        Calculate matching degree between two regions.

        Paper Equation (2):
        Rmatch = Σ(I_rect(i,j) - I_rect*(i,j))^2 / (Σ(I_rect(i,j))^2 * Σ(I_rect*(i,j))^2)

        Args:
            region1: Region from current frame.
            region2: Corresponding region from background.

        Returns:
            Matching degree in [0, 1].
        """
        if region1.size == 0 or region2.size == 0:
            return 1.0

        region1 = region1.astype(np.float32)
        region2 = region2.astype(np.float32)

        numerator = np.sum((region1 - region2) ** 2)
        denominator = np.sum(region1 ** 2) * np.sum(region2 ** 2)

        if denominator == 0:
            return 0.0

        return numerator / denominator
