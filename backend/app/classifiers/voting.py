"""
SVM-Based Classification with Voting.

Paper Section: 2.4 Classification and decision making

Implements majority voting using 3 independent SVM classifiers:
- SVM-ART: trained on artificial features
- SVM-GLCM: trained on GLCM features
- SVM-DWT: trained on DWT features

The final decision is made by majority voting among the three classifiers.

Training on:
- 1000 smoky vehicle samples
- 1000 non-smoky vehicle samples
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class SVMVotingClassifier:
    """SVM voting classifier for smoke detection."""

    def __init__(self):
        """Initialize voting classifier."""
        self.svm_art: Optional[SVC] = None
        self.svm_glcm: Optional[SVC] = None
        self.svm_dwt: Optional[SVC] = None
        self.scaler_art = None
        self.scaler_glcm = None
        self.scaler_dwt = None

    def train(
        self,
        X_art: np.ndarray,
        X_glcm: np.ndarray,
        X_dwt: np.ndarray,
        y: np.ndarray,
    ) -> bool:
        """
        Train 3 independent SVM classifiers.

        Args:
            X_art: ART features (n_samples, 4).
            X_glcm: GLCM features (n_samples, 20).
            X_dwt: DWT features (n_samples, 12).
            y: Labels (n_samples,) - 1 for smoky, 0 for non-smoky.

        Returns:
            True if training successful, False otherwise.
        """
        try:
            from sklearn.preprocessing import StandardScaler

            # Normalize features
            self.scaler_art = StandardScaler()
            X_art_scaled = self.scaler_art.fit_transform(X_art)

            self.scaler_glcm = StandardScaler()
            X_glcm_scaled = self.scaler_glcm.fit_transform(X_glcm)

            self.scaler_dwt = StandardScaler()
            X_dwt_scaled = self.scaler_dwt.fit_transform(X_dwt)

            # Train SVM classifiers with RBF kernel
            self.svm_art = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            self.svm_art.fit(X_art_scaled, y)

            self.svm_glcm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            self.svm_glcm.fit(X_glcm_scaled, y)

            self.svm_dwt = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            self.svm_dwt.fit(X_dwt_scaled, y)

            logger.info("SVM models trained successfully")
            return True

        except Exception as e:
            logger.error(f"Error training SVM models: {e}")
            return False

    def predict(
        self,
        features: dict,
    ) -> tuple[str, float, dict]:
        """
        Predict class using majority voting.

        Args:
            features: Dictionary of extracted features with keys:
                - art_* (4 features)
                - glcm_* (20 features)
                - dwt_* (12 features)

        Returns:
            Tuple of (class_name, confidence, voting_details).
        """
        if self.svm_art is None or self.svm_glcm is None or self.svm_dwt is None:
            logger.warning("SVM models not trained, using heuristic classification")
            return self._heuristic_predict(features)

        try:
            # Extract feature vectors
            art_features = self._extract_art_features(features)
            glcm_features = self._extract_glcm_features(features)
            dwt_features = self._extract_dwt_features(features)

            if (art_features is None or glcm_features is None or dwt_features is None):
                return self._heuristic_predict(features)

            # Normalize features
            art_scaled = self.scaler_art.transform(art_features.reshape(1, -1))
            glcm_scaled = self.scaler_glcm.transform(glcm_features.reshape(1, -1))
            dwt_scaled = self.scaler_dwt.transform(dwt_features.reshape(1, -1))

            # Get predictions
            pred_art = self.svm_art.predict(art_scaled)[0]
            pred_glcm = self.svm_glcm.predict(glcm_scaled)[0]
            pred_dwt = self.svm_dwt.predict(dwt_scaled)[0]

            # Get probabilities
            prob_art = self.svm_art.predict_proba(art_scaled)[0]
            prob_glcm = self.svm_glcm.predict_proba(glcm_scaled)[0]
            prob_dwt = self.svm_dwt.predict_proba(dwt_scaled)[0]

            # Majority voting
            votes = np.array([pred_art, pred_glcm, pred_dwt])
            smoky_votes = np.sum(votes == 1)
            non_smoky_votes = np.sum(votes == 0)

            if smoky_votes > non_smoky_votes:
                predicted_class = 1
                # Average probability for smoky class
                confidence = np.mean([prob_art[1], prob_glcm[1], prob_dwt[1]])
            else:
                predicted_class = 0
                # Average probability for non-smoky class
                confidence = np.mean([prob_art[0], prob_glcm[0], prob_dwt[0]])

            class_name = "smoky" if predicted_class == 1 else "non_smoky"

            voting_details = {
                "art_prediction": "smoky" if pred_art == 1 else "non_smoky",
                "art_confidence": float(prob_art[1]),
                "glcm_prediction": "smoky" if pred_glcm == 1 else "non_smoky",
                "glcm_confidence": float(prob_glcm[1]),
                "dwt_prediction": "smoky" if pred_dwt == 1 else "non_smoky",
                "dwt_confidence": float(prob_dwt[1]),
                "smoky_votes": int(smoky_votes),
                "non_smoky_votes": int(non_smoky_votes),
            }

            return class_name, float(confidence), voting_details

        except Exception as e:
            logger.error(f"Error in SVM prediction: {e}")
            return self._heuristic_predict(features)

    @staticmethod
    def _extract_art_features(features: dict) -> Optional[np.ndarray]:
        """Extract ART feature vector."""
        try:
            art_order = ['art_rate', 'art_match', 'art_mean', 'art_variance']
            art_values = [features.get(key, 0.0) for key in art_order]
            return np.array(art_values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error extracting ART features: {e}")
            return None

    @staticmethod
    def _extract_glcm_features(features: dict) -> Optional[np.ndarray]:
        """Extract GLCM feature vector."""
        try:
            glcm_keys = []
            for prop in ['asm', 'ent', 'con', 'cor', 'idm']:
                for angle in ['0', '45', '90', '135']:
                    glcm_keys.append(f'glcm_{prop}_{angle}')
            
            glcm_values = [features.get(key, 0.0) for key in glcm_keys]
            return np.array(glcm_values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error extracting GLCM features: {e}")
            return None

    @staticmethod
    def _extract_dwt_features(features: dict) -> Optional[np.ndarray]:
        """Extract DWT feature vector."""
        try:
            dwt_keys = []
            for level in ['l1', 'l2']:
                for direction in ['h', 'v', 'd']:
                    for block in ['b1', 'b2']:
                        dwt_keys.append(f'dwt_energy_{level}_{direction}_{block}')
            
            dwt_values = [features.get(key, 0.0) for key in dwt_keys]
            return np.array(dwt_values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error extracting DWT features: {e}")
            return None

    @staticmethod
    def _heuristic_predict(features: dict) -> tuple[str, float, dict]:
        """
        Fallback heuristic prediction when SVM models not available.

        Args:
            features: Dictionary of extracted features.

        Returns:
            Tuple of (class_name, confidence, voting_details).
        """
        try:
            art_mean = features.get('art_mean', 0.0)
            glcm_ent_0 = features.get('glcm_ent_0', 0.0)
            dwt_l1_h_b1 = features.get('dwt_energy_l1_h_b1', 0.0)

            # Smoke likelihood score (0-1)
            score = (art_mean / 50.0) + (glcm_ent_0 / 10.0) + (dwt_l1_h_b1 / 1000.0)
            score = min(1.0, max(0.0, score / 3.0))

            if score > 0.5:
                return 'smoky', score, {"heuristic": True}
            else:
                return 'non_smoky', 1.0 - score, {"heuristic": True}

        except Exception as e:
            logger.error(f"Error in heuristic prediction: {e}")
            return 'non_smoky', 0.5, {"heuristic": True, "error": True}
