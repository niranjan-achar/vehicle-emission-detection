"""
Key Region (Smoke-Relevant Region) Extraction.

Paper Section: 2.3 Feature extraction

The key region is the rectangular area behind the vehicle rear where smoke
is most likely to appear. Dimensions:
- Width: 0.7 * vehicle_width (centered on rear)
- Height: ε = 10 pixels
- Position: centered horizontally on vehicle, starting at rear_y - 5 to rear_y + 5
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KeyRegionExtractor:
    """Extract key region (smoke area) from vehicle."""

    def __init__(self, width_ratio: float = 0.7, height: int = 10, margin: int = 5):
        """
        Initialize key region extractor.

        Args:
            width_ratio: Width as ratio of vehicle width (default 0.7).
            height: Height in pixels above rear position (default 10).
            margin: Margin above and below rear position (default 5).
        """
        self.width_ratio = width_ratio
        self.height = height
        self.margin = margin

    def extract_key_region(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        rear_y: int,
    ) -> tuple[np.ndarray, dict]:
        """
        Extract key region from frame.

        Paper Section 2.3:
        - Region centered on vehicle
        - Width = width_ratio * vehicle_width
        - Height = height (typically 10 pixels)
        - Position: around rear_y ± margin

        Args:
            frame: Input frame (H, W, 3) BGR or (H, W) grayscale.
            bbox: Vehicle bounding box (x, y, w, h).
            rear_y: Vehicle rear y-coordinate.

        Returns:
            Tuple of (key_region, metadata):
            - key_region: Extracted region (height, width_ratio*vehicle_width)
            - metadata: dict with region coordinates and dimensions
        """
        x, y, w, h = bbox

        if w == 0:
            return None, None

        # Calculate key region dimensions
        region_width = max(1, int(w * self.width_ratio))
        region_height = self.height + 2 * self.margin

        # Center horizontally on vehicle
        region_x1 = x + (w - region_width) // 2
        region_x2 = region_x1 + region_width

        # Position vertically around rear_y
        region_y1 = rear_y - self.margin
        region_y2 = region_y1 + region_height

        # Clamp to frame boundaries
        h_frame, w_frame = frame.shape[:2]
        region_x1 = max(0, min(region_x1, w_frame - 1))
        region_x2 = max(region_x1 + 1, min(region_x2, w_frame))
        region_y1 = max(0, min(region_y1, h_frame - 1))
        region_y2 = max(region_y1 + 1, min(region_y2, h_frame))

        # Extract region
        if len(frame.shape) == 3:
            # Multi-channel frame, extract from all channels
            key_region = frame[region_y1:region_y2, region_x1:region_x2, :].copy()
        else:
            # Grayscale frame
            key_region = frame[region_y1:region_y2, region_x1:region_x2].copy()

        if key_region.size == 0:
            return None, None

        metadata = {
            "region_x1": region_x1,
            "region_x2": region_x2,
            "region_y1": region_y1,
            "region_y2": region_y2,
            "region_width": region_x2 - region_x1,
            "region_height": region_y2 - region_y1,
            "vehicle_bbox": bbox,
            "rear_y": rear_y,
        }

        logger.debug(
            f"Extracted key region: ({region_x1}, {region_y1}) to "
            f"({region_x2}, {region_y2})"
        )

        return key_region, metadata

    def visualize_key_region(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        rear_y: int,
    ) -> np.ndarray:
        """
        Visualize key region on frame (for debugging).

        Args:
            frame: Input frame (H, W, 3) BGR.
            bbox: Vehicle bounding box (x, y, w, h).
            rear_y: Vehicle rear y-coordinate.

        Returns:
            Frame with key region highlighted.
        """
        if len(frame.shape) != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_vis = frame.copy()

        x, y, w, h = bbox

        # Draw vehicle bounding box
        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw key region
        region_width = max(1, int(w * self.width_ratio))
        region_height = self.height + 2 * self.margin
        region_x1 = x + (w - region_width) // 2
        region_y1 = rear_y - self.margin
        region_x2 = region_x1 + region_width
        region_y2 = region_y1 + region_height

        h_frame, w_frame = frame_vis.shape[:2]
        region_x1 = max(0, min(region_x1, w_frame - 1))
        region_x2 = max(region_x1 + 1, min(region_x2, w_frame))
        region_y1 = max(0, min(region_y1, h_frame - 1))
        region_y2 = max(region_y1 + 1, min(region_y2, h_frame))

        cv2.rectangle(frame_vis, (region_x1, region_y1), (region_x2, region_y2), (0, 0, 255), 2)

        # Mark rear position
        cv2.line(
            frame_vis, (region_x1, rear_y), (region_x2, rear_y), (255, 0, 0), 1
        )

        return frame_vis
