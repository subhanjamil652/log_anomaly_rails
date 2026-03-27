"""Isolation Forest unsupervised anomaly detector for BGL log data."""

import os
import logging
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """
    Isolation Forest unsupervised anomaly detector.
    Does not require labelled training data — detects anomalies via
    random isolation path length in an ensemble of trees.
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.08,
                 random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        )
        self.name = "Isolation Forest"
        self.model_type = "unsupervised"
        self._score_min = None
        self._score_max = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "IsolationForestModel":
        """Unsupervised fit — labels ignored."""
        logger.info(f"Training {self.name} on {X.shape[0]:,} samples ...")
        self.model.fit(X)
        scores = self.model.score_samples(X)
        self._score_min = float(scores.min())
        self._score_max = float(scores.max())
        logger.info(f"{self.name} training complete. "
                    f"Score range: [{self._score_min:.4f}, {self._score_max:.4f}]")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 (anomaly) or 0 (normal)."""
        raw = self.model.predict(X)   # sklearn: -1=anomaly, 1=normal
        return np.where(raw == -1, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly probability in [0, 1] (higher = more anomalous)."""
        scores = self.model.score_samples(X)   # lower = more anomalous
        lo = self._score_min or scores.min()
        hi = self._score_max or scores.max()
        if hi == lo:
            return np.zeros(len(scores))
        # invert so higher value = higher anomaly probability
        proba = 1.0 - (scores - lo) / (hi - lo)
        return np.clip(proba, 0.0, 1.0)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"{self.name} saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IsolationForestModel":
        obj = joblib.load(path)
        logger.info(f"IsolationForestModel loaded from {path}")
        return obj
