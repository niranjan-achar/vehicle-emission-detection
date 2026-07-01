"""
Background Subtraction using ViBe Algorithm.

Paper Section: 2.1 Vehicle detection
Implementation: ViBe (Visual Background Extractor) for foreground detection.

Reference:
Barnich, O., Van Droogenbroeck, M.: 'Vibe: a universal background
subtraction algorithm for video sequences', IEEE Trans. Image Process.,
2011, 20, (6), pp. 1709–1724

The ViBe algorithm maintains a set of sample backgrounds for each pixel
and uses Euclidean distance to classify pixels as foreground or background.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundSubtractor:
    """ViBe-based background subtraction for vehicle detection."""

    def __init__(self, samples: int = 20, min_dist: int = 20, req_matches: int = 2):
        """
        Initialize ViBe background subtractor.

        Args:
            samples: Number of samples per pixel.
            min_dist: Minimum distance threshold for background model.
            req_matches: Required matches to update background model.
        """
        self.samples = samples
        self.min_dist = min_dist
        self.req_matches = req_matches
        self.bg_model: Optional[dict] = None
        self.frame_count = 0

    def apply(self, frame: np.ndarray, learn_rate: float = 0.003) -> np.ndarray:
        """
        Apply ViBe background subtraction.

        Args:
            frame: Input frame (H, W, 3) BGR.
            learn_rate: Learning rate for model update.

        Returns:
            Foreground mask (H, W) where 255=foreground, 0=background.
        """
        if frame is None or frame.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape

        if self.bg_model is None:
            self._init_model(gray)

        foreground = self._segment(gray)
        self._update_model(gray, foreground, learn_rate)
        self.frame_count += 1

        return foreground

    def _init_model(self, frame: np.ndarray) -> None:
        """Initialize ViBe model with first frame."""
        h, w = frame.shape
        self.bg_model = {}

        for y in range(h):
            for x in range(w):
                pixel_val = int(frame[y, x])
                # Initialize with samples from neighborhood
                samples = []
                for _ in range(self.samples):
                    ny = max(0, min(h - 1, y + random.randint(-1, 1)))
                    nx = max(0, min(w - 1, x + random.randint(-1, 1)))
                    samples.append(int(frame[ny, nx]))
                self.bg_model[(y, x)] = samples

    def _segment(self, frame: np.ndarray) -> np.ndarray:
        """Segment foreground using ViBe model."""
        h, w = frame.shape
        foreground = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                pixel_val = int(frame[y, x])
                samples = self.bg_model.get((y, x), [])

                if not samples:
                    foreground[y, x] = 0
                    continue

                # Count matches within min_dist
                matches = sum(1 for s in samples if abs(int(s) - pixel_val) < self.min_dist)

                if matches < self.req_matches:
                    foreground[y, x] = 255
                else:
                    foreground[y, x] = 0

        return foreground

    def _update_model(self, frame: np.ndarray, foreground: np.ndarray, learn_rate: float) -> None:
        """Update ViBe model with new frame."""
        h, w = frame.shape

        for y in range(h):
            for x in range(w):
                if (y, x) not in self.bg_model:
                    continue

                pixel_val = int(frame[y, x])

                # Update only background pixels (stochastically)
                if foreground[y, x] == 0 and random.random() < learn_rate:
                    sample_idx = random.randint(0, self.samples - 1)
                    self.bg_model[(y, x)][sample_idx] = pixel_val

                    # Update neighbor samples
                    if random.random() < learn_rate:
                        ny = max(0, min(h - 1, y + random.randint(-1, 1)))
                        nx = max(0, min(w - 1, x + random.randint(-1, 1)))
                        if (ny, nx) in self.bg_model:
                            neighbor_idx = random.randint(0, self.samples - 1)
                            self.bg_model[(ny, nx)][neighbor_idx] = pixel_val
