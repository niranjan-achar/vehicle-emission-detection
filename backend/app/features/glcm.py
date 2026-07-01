"""
GLCM Features for Smoke Detection.

Paper Section: 2.3 Feature extraction, subsection "GLCM features"

Gray-Level Co-occurrence Matrix (GLCM) features capture texture patterns
in the key region. The paper extracts 5 statistics at 4 orientations (θ):
- Angular Second Moment (ASM)
- Entropy (ENT)
- Contrast (CON)
- Correlation (COR)
- Inverse Difference Moment (IDM)

With d=2 pixels and θ ∈ {0°, 45°, 90°, 135°}, total features: 5 × 4 = 20

Reference:
Haralick, R. M., Shanmugam, K., Dinstein, I. (1973).
'Textural features for image classification',
IEEE Trans. Syst., Man, Cybern., 3, (6), pp. 610–621
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.feature import graycomatrix, graycoprops

logger = logging.getLogger(__name__)


class GLCMFeatureExtractor:
    """Extract GLCM (texture) features from key region."""

    def __init__(self, distance: int = 2, levels: int = 256):
        """
        Initialize GLCM feature extractor.

        Args:
            distance: Pixel distance for GLCM (default 2).
            levels: Number of quantization levels (default 256 for uint8).
        """
        self.distance = distance
        self.levels = levels

    def extract_features(self, key_region: np.ndarray) -> dict:
        """
        Extract GLCM features at 4 orientations.

        Paper specifies:
        - d = 2 pixels
        - θ ∈ {0°, 45°, 90°, 135°} = {0, 45, 90, 135}
        - 5 texture properties: ASM, ENT, CON, COR, IDM

        Args:
            key_region: Input key region (H, W, 3) BGR or (H, W) grayscale.

        Returns:
            Dictionary with 20 GLCM features:
            - glcm_asm_0, glcm_asm_45, glcm_asm_90, glcm_asm_135
            - glcm_ent_0, glcm_ent_45, glcm_ent_90, glcm_ent_135
            - glcm_con_0, glcm_con_45, glcm_con_90, glcm_con_135
            - glcm_cor_0, glcm_cor_45, glcm_cor_90, glcm_cor_135
            - glcm_idm_0, glcm_idm_45, glcm_idm_90, glcm_idm_135
        """
        if key_region is None or key_region.size == 0:
            return self._default_features()

        # Convert to grayscale if needed
        if len(key_region.shape) == 3:
            import cv2
            gray_region = cv2.cvtColor(key_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = key_region.astype(np.uint8)

        # Quantize to 256 levels if needed
        if gray_region.dtype != np.uint8:
            gray_region = (gray_region / gray_region.max() * 255).astype(np.uint8)

        # Compute GLCM at 4 orientations
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
        angle_names = ['0', '45', '90', '135']

        glcm = graycomatrix(
            gray_region,
            distances=[self.distance],
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        features = {}

        asm_values = graycoprops(glcm, 'ASM')[0]
        con_values = graycoprops(glcm, 'contrast')[0]
        cor_values = graycoprops(glcm, 'correlation')[0]
        idm_values = graycoprops(glcm, 'homogeneity')[0]

        ent_values = []
        for angle_idx in range(len(angle_names)):
            glcm_slice = glcm[:, :, 0, angle_idx]
            entropy = -np.sum(glcm_slice * np.log2(glcm_slice + 1e-12))
            ent_values.append(float(entropy))

        value_map = {
            'asm': asm_values,
            'ent': ent_values,
            'con': con_values,
            'cor': cor_values,
            'idm': idm_values,
        }

        for prop_name in ['asm', 'ent', 'con', 'cor', 'idm']:
            prop_values = value_map[prop_name]
            for angle_idx, angle_name in enumerate(angle_names):
                feature_name = f'glcm_{prop_name}_{angle_name}'
                features[feature_name] = float(prop_values[angle_idx])

        logger.debug(f"Extracted {len(features)} GLCM features")

        return features if len(features) == 20 else self._default_features()

    @staticmethod
    def _default_features() -> dict:
        """Return default GLCM features (all zeros)."""
        features = {}
        property_names = ['asm', 'ent', 'con', 'cor', 'idm']
        angle_names = ['0', '45', '90', '135']

        for prop in property_names:
            for angle in angle_names:
                features[f'glcm_{prop}_{angle}'] = 0.0

        return features


class GLCMFeatureExtractorLegacy:
    """Legacy GLCM extractor without scikit-image dependency (fallback)."""

    def __init__(self, distance: int = 2, levels: int = 256):
        """Initialize GLCM feature extractor."""
        self.distance = distance
        self.levels = levels

    def extract_features(self, key_region: np.ndarray) -> dict:
        """Extract GLCM features (approximation without scikit-image)."""
        if key_region is None or key_region.size == 0:
            return self._default_features()

        # Convert to grayscale
        if len(key_region.shape) == 3:
            import cv2
            gray_region = cv2.cvtColor(key_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = key_region.astype(np.uint8)

        features = {}

        # Compute texture features using edge detection and gradients as approximation
        import cv2

        # Approximation: Use Laplacian and Sobel as texture descriptors
        laplacian = cv2.Laplacian(gray_region, cv2.CV_64F)
        sobelx = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)

        # Generate 20 features from texture descriptors
        property_names = ['asm', 'ent', 'con', 'cor', 'idm']
        angle_names = ['0', '45', '90', '135']

        for prop_idx, prop_name in enumerate(property_names):
            for angle_idx, angle_name in enumerate(angle_names):
                feature_name = f'glcm_{prop_name}_{angle_name}'
                
                # Approximate values from texture descriptors
                if prop_idx == 0:  # asm - energy
                    features[feature_name] = float(np.mean(laplacian**2) / 10000)
                elif prop_idx == 1:  # ent - entropy
                    features[feature_name] = float(np.std(laplacian) / 50)
                elif prop_idx == 2:  # con - contrast
                    features[feature_name] = float(np.mean(np.abs(laplacian)) / 50)
                elif prop_idx == 3:  # cor - correlation
                    sx_mean = np.mean(sobelx)
                    sy_mean = np.mean(sobely)
                    sx_var = np.var(sobelx)
                    sy_var = np.var(sobely)
                    if sx_var > 0 and sy_var > 0:
                        features[feature_name] = float(
                            np.mean((sobelx - sx_mean) * (sobely - sy_mean)) / 
                            np.sqrt(sx_var * sy_var)
                        )
                    else:
                        features[feature_name] = 0.0
                else:  # idm - homogeneity
                    features[feature_name] = float(1.0 / (1.0 + np.mean(np.abs(laplacian))))

        return features

    @staticmethod
    def _default_features() -> dict:
        """Return default GLCM features."""
        features = {}
        property_names = ['asm', 'ent', 'con', 'cor', 'idm']
        angle_names = ['0', '45', '90', '135']

        for prop in property_names:
            for angle in angle_names:
                features[f'glcm_{prop}_{angle}'] = 0.0

        return features
