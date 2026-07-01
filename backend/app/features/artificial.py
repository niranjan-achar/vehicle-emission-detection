"""
Artificial Features (ART) for Smoke Detection.

Paper Section: 2.3 Feature extraction, subsection "Artificial features"

The ART features are extracted from the vehicle's rear region using a
reference non-smoke image. Four features are computed:
1. ARTrate: Ratio of changed pixels
2. ARTmatch: Template matching coefficient
3. ARTmean: Mean difference
4. ARTvariance: Variance of differences

These features capture localized intensity changes typical of smoke.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ArtificialFeatureExtractor:
    """Extract artificial (ART) features from key region."""

    def __init__(self):
        """Initialize ART feature extractor."""
        pass

    def extract_features(
        self,
        key_region: np.ndarray,
        reference_region: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Extract four ART features from key region.

        Paper Equations for ART features:
        - ARTrate: Ratio of pixels with intensity change > threshold
        - ARTmatch: Template matching correlation
        - ARTmean: Mean absolute difference
        - ARTvariance: Variance of differences

        Args:
            key_region: Input key region (H, W, 3) BGR or (H, W) grayscale.
            reference_region: Reference non-smoke region for comparison.
                If None, uses internal reference.

        Returns:
            Dictionary with ART features:
            - 'art_rate': float in [0, 1]
            - 'art_match': float in [0, 1]
            - 'art_mean': float >= 0
            - 'art_variance': float >= 0
        """
        if key_region is None or key_region.size == 0:
            return {
                'art_rate': 0.0,
                'art_match': 0.0,
                'art_mean': 0.0,
                'art_variance': 0.0,
            }

        # Convert to grayscale if needed
        if len(key_region.shape) == 3:
            gray_region = cv2.cvtColor(key_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = key_region.copy()

        # Use reference if provided, otherwise use self-reference
        if reference_region is None:
            # Use Gaussian smoothed version as reference (approximate non-smoke)
            reference_region = cv2.GaussianBlur(gray_region, (5, 5), 1.0)
        else:
            if len(reference_region.shape) == 3:
                reference_region = cv2.cvtColor(reference_region, cv2.COLOR_BGR2GRAY)

        # Ensure same size
        if gray_region.shape != reference_region.shape:
            reference_region = cv2.resize(reference_region, (gray_region.shape[1], gray_region.shape[0]))

        features = {}

        # Feature 1: ARTrate (ratio of changed pixels)
        features['art_rate'] = self._calc_art_rate(gray_region, reference_region)

        # Feature 2: ARTmatch (template matching correlation)
        features['art_match'] = self._calc_art_match(gray_region, reference_region)

        # Feature 3: ARTmean (mean difference)
        features['art_mean'] = self._calc_art_mean(gray_region, reference_region)

        # Feature 4: ARTvariance (variance of differences)
        features['art_variance'] = self._calc_art_variance(gray_region, reference_region)

        logger.debug(f"Extracted ART features: {features}")

        return features

    @staticmethod
    def _calc_art_rate(current: np.ndarray, reference: np.ndarray, threshold: int = 20) -> float:
        """
        Calculate ARTrate: Ratio of pixels with intensity change > threshold.

        Args:
            current: Current frame region.
            reference: Reference frame region.
            threshold: Intensity change threshold.

        Returns:
            ARTrate in [0, 1].
        """
        diff = np.abs(current.astype(np.float32) - reference.astype(np.float32))
        changed_pixels = np.sum(diff > threshold)
        total_pixels = current.size

        if total_pixels == 0:
            return 0.0

        art_rate = changed_pixels / total_pixels
        return float(np.clip(art_rate, 0.0, 1.0))

    @staticmethod
    def _calc_art_match(current: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate ARTmatch: Normalized cross-correlation.

        Args:
            current: Current frame region.
            reference: Reference frame region.

        Returns:
            ARTmatch in [0, 1].
        """
        current_f = current.astype(np.float32)
        reference_f = reference.astype(np.float32)

        # Normalize
        current_f = (current_f - np.mean(current_f)) / (np.std(current_f) + 1e-6)
        reference_f = (reference_f - np.mean(reference_f)) / (np.std(reference_f) + 1e-6)

        # Compute normalized cross-correlation
        correlation = np.mean(current_f * reference_f)

        # Scale to [0, 1]
        art_match = (1.0 + correlation) / 2.0

        return float(np.clip(art_match, 0.0, 1.0))

    @staticmethod
    def _calc_art_mean(current: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate ARTmean: Mean absolute difference.

        Args:
            current: Current frame region.
            reference: Reference frame region.

        Returns:
            ARTmean >= 0.
        """
        diff = np.abs(current.astype(np.float32) - reference.astype(np.float32))
        art_mean = np.mean(diff)

        return float(art_mean)

    @staticmethod
    def _calc_art_variance(current: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate ARTvariance: Variance of differences.

        Args:
            current: Current frame region.
            reference: Reference frame region.

        Returns:
            ARTvariance >= 0.
        """
        diff = np.abs(current.astype(np.float32) - reference.astype(np.float32))
        art_variance = np.var(diff)

        return float(art_variance)
