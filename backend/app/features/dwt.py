"""
DWT Features for Smoke Detection.

Paper Section: 2.3 Feature extraction, subsection "DWT features"

Discrete Wavelet Transform features capture multi-resolution texture patterns.
The paper uses:
- Wavelet: Daubechies (db1) or Haar
- 2 decomposition levels
- 3 directional subbands per level: horizontal, vertical, diagonal
- 2 blocks per subband
- Total features: 2 levels × 3 directions × 2 blocks = 12 features

Features extracted: Energy (sum of absolute coefficients) for each subband block.

Reference:
Mallat, S.: 'A theory for multiresolution signal decomposition: the wavelet
representation', IEEE Trans. Pattern Anal. Mach. Intell., 1989, 11, (7), pp. 674–693
"""

from __future__ import annotations

import logging

import numpy as np
import pywt

logger = logging.getLogger(__name__)


class DWTFeatureExtractor:
    """Extract DWT (wavelet) features from key region."""

    def __init__(self, wavelet: str = 'db1', levels: int = 2):
        """
        Initialize DWT feature extractor.

        Args:
            wavelet: Wavelet name (default 'db1' for Daubechies-1).
            levels: Decomposition levels (default 2).
        """
        self.wavelet = wavelet
        self.levels = levels

    def extract_features(self, key_region: np.ndarray) -> dict:
        """
        Extract DWT features at multiple levels.

        Paper specifies:
        - 2 decomposition levels
        - 3 orientations per level: horizontal (cH), vertical (cV), diagonal (cD)
        - 2 blocks per subband
        - Feature = energy (sum of absolute values) per block

        Args:
            key_region: Input key region (H, W, 3) BGR or (H, W) grayscale.

        Returns:
            Dictionary with 12 DWT features:
            - dwt_energy_l1_h_b1, dwt_energy_l1_h_b2 (level 1 horizontal)
            - dwt_energy_l1_v_b1, dwt_energy_l1_v_b2 (level 1 vertical)
            - dwt_energy_l1_d_b1, dwt_energy_l1_d_b2 (level 1 diagonal)
            - dwt_energy_l2_h_b1, dwt_energy_l2_h_b2 (level 2 horizontal)
            - dwt_energy_l2_v_b1, dwt_energy_l2_v_b2 (level 2 vertical)
            - dwt_energy_l2_d_b1, dwt_energy_l2_d_b2 (level 2 diagonal)
        """
        if key_region is None or key_region.size == 0:
            return self._default_features()

        # Convert to grayscale if needed
        if len(key_region.shape) == 3:
            import cv2
            gray_region = cv2.cvtColor(key_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = key_region.astype(np.float32)

        # Normalize to [0, 1]
        if gray_region.max() > 1.0:
            gray_region = gray_region / 255.0

        features = {}

        try:
            # Decompose at level 1
            cA1, (cH1, cV1, cD1) = pywt.dwt2(gray_region, self.wavelet)

            # Extract features from level 1
            features['dwt_energy_l1_h_b1'] = self._calc_block_energy(cH1, block_idx=0)
            features['dwt_energy_l1_h_b2'] = self._calc_block_energy(cH1, block_idx=1)
            features['dwt_energy_l1_v_b1'] = self._calc_block_energy(cV1, block_idx=0)
            features['dwt_energy_l1_v_b2'] = self._calc_block_energy(cV1, block_idx=1)
            features['dwt_energy_l1_d_b1'] = self._calc_block_energy(cD1, block_idx=0)
            features['dwt_energy_l1_d_b2'] = self._calc_block_energy(cD1, block_idx=1)

            # Decompose at level 2
            cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, self.wavelet)

            # Extract features from level 2
            features['dwt_energy_l2_h_b1'] = self._calc_block_energy(cH2, block_idx=0)
            features['dwt_energy_l2_h_b2'] = self._calc_block_energy(cH2, block_idx=1)
            features['dwt_energy_l2_v_b1'] = self._calc_block_energy(cV2, block_idx=0)
            features['dwt_energy_l2_v_b2'] = self._calc_block_energy(cV2, block_idx=1)
            features['dwt_energy_l2_d_b1'] = self._calc_block_energy(cD2, block_idx=0)
            features['dwt_energy_l2_d_b2'] = self._calc_block_energy(cD2, block_idx=1)

            logger.debug(f"Extracted {len(features)} DWT features")

        except Exception as e:
            logger.error(f"Error extracting DWT features: {e}")
            return self._default_features()

        return features if len(features) == 12 else self._default_features()

    @staticmethod
    def _calc_block_energy(coeff: np.ndarray, block_idx: int = 0) -> float:
        """
        Calculate energy (sum of absolute values) for a block of coefficients.

        Paper: Energy = Σ|coefficient|

        Args:
            coeff: Wavelet coefficient matrix.
            block_idx: Block index (0 = upper half, 1 = lower half).

        Returns:
            Energy value >= 0.
        """
        if coeff is None or coeff.size == 0:
            return 0.0

        # Split into 2 blocks (upper and lower halves)
        h = coeff.shape[0]
        mid = h // 2

        if block_idx == 0:
            block = coeff[:mid, :]
        else:
            block = coeff[mid:, :]

        # Calculate energy
        energy = np.sum(np.abs(block.astype(np.float32)))

        return float(energy)

    @staticmethod
    def _default_features() -> dict:
        """Return default DWT features (all zeros)."""
        features = {}
        levels = ['l1', 'l2']
        directions = ['h', 'v', 'd']
        blocks = ['b1', 'b2']

        for level in levels:
            for direction in directions:
                for block in blocks:
                    features[f'dwt_energy_{level}_{direction}_{block}'] = 0.0

        return features
