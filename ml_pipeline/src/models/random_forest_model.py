"""Random Forest classifier for BGL log anomaly detection."""

import os
import logging
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest ensemble classifier.
    Primary candidate model — expected to achieve F1 > 0.90 on BGL.
    Uses class_weight='balanced_subsample' for robust imbalance handling.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = None,
                 min_samples_leaf: int = 2, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
            oob_score=True,
        )
        self.name = "Random Forest"
        self.model_type = "supervised"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        logger.info(f"Training {self.name} on {X.shape[0]:,} samples, "
                    f"{X.shape[1]} features ...")
        self.model.fit(X, y)
        logger.info(f"{self.name} OOB score: {self.model.oob_score_:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"{self.name} saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        obj = joblib.load(path)
        logger.info(f"RandomForestModel loaded from {path}")
        return obj
