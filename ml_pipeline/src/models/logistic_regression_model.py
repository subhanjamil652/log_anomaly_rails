"""Logistic Regression baseline model for BGL log anomaly detection."""

import os
import logging
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """
    Logistic Regression baseline classifier.
    Uses class_weight='balanced' to handle the 1:12.6 BGL class imbalance.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        )
        self.name = "Logistic Regression"
        self.model_type = "supervised"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        logger.info(f"Training {self.name} on {X.shape[0]:,} samples ...")
        self.model.fit(X, y)
        logger.info(f"{self.name} training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"{self.name} saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        obj = joblib.load(path)
        logger.info(f"LogisticRegressionModel loaded from {path}")
        return obj
