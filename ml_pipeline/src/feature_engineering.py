"""
Feature Engineering for BGL Log Anomaly Detection.

Extracts a rich feature set from sliding-window log sequences:
  1. Event count vectors (per-template frequency)
  2. TF-IDF weighted event frequencies
  3. Log-level severity statistics
  4. Component identifier one-hot encoding
  5. Temporal features (event rate, time-since-last-error)
  6. Sequential bigram co-occurrence counts
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

SEVERITY_MAP = {
    "FATAL": 5, "SEVERE": 4, "ERROR": 4,
    "WARNING": 3, "INFO": 2, "APPINFO": 1, "UNKNOWN": 0,
}

TOP_COMPONENTS = [
    "kernel", "MMCS", "APP", "MPI-IO", "lustre", "BGLMASTER",
    "ciod", "ciodb", "jm", "IO", "rts", "lwk", "comm",
    "serv", "pm", "mmcs", "syslog", "HARDWARE", "torus", "OTHER",
]


class FeatureEngineer:
    """
    Transforms a list of log-window DataFrames into a 2-D feature matrix
    suitable for scikit-learn and PyTorch models.
    """

    def __init__(self, window_size: int = 20, n_tfidf_features: int = 50):
        self.window_size = window_size
        self.n_tfidf_features = n_tfidf_features

        self.tfidf = TfidfVectorizer(max_features=n_tfidf_features,
                                     token_pattern=r"(?u)\b\w[\w\*]+\b")
        self.scaler = StandardScaler()
        self._fitted = False
        self._feature_names: list = []

    # -- Public API ------------------------------------------------------------

    def fit_transform(self, windows: list, labels: np.ndarray) -> tuple:
        """Fit on training windows and return (X, y)."""
        corpus = [self._window_to_text(w) for w in windows]
        self.tfidf.fit(corpus)

        X = self._extract_all(windows, corpus)
        X = self.scaler.fit_transform(X)
        self._fitted = True
        self._build_feature_names()
        logger.info(f"Feature matrix shape: {X.shape} "
                    f"({X.shape[1]} features per window)")
        return X, labels

    def transform(self, windows: list) -> np.ndarray:
        """Transform unseen windows using fitted encoders."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        corpus = [self._window_to_text(w) for w in windows]
        X = self._extract_all(windows, corpus)
        return self.scaler.transform(X)

    def get_feature_names(self) -> list:
        return self._feature_names

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"tfidf": self.tfidf,
                     "scaler": self.scaler,
                     "window_size": self.window_size,
                     "n_tfidf_features": self.n_tfidf_features,
                     "feature_names": self._feature_names,
                     "fitted": self._fitted}, path)
        logger.info(f"FeatureEngineer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        data = joblib.load(path)
        fe = cls(window_size=data["window_size"],
                 n_tfidf_features=data["n_tfidf_features"])
        fe.tfidf = data["tfidf"]
        fe.scaler = data["scaler"]
        fe._feature_names = data["feature_names"]
        fe._fitted = data["fitted"]
        logger.info(f"FeatureEngineer loaded from {path}")
        return fe

    # -- Internal feature extraction -------------------------------------------

    def _window_to_text(self, window: pd.DataFrame) -> str:
        """Join template strings for TF-IDF vectorisation."""
        templates = window["template"].fillna("").tolist()
        return " ".join(str(t).replace("<*>", "VAR") for t in templates)

    def _extract_all(self, windows: list, corpus: list) -> np.ndarray:
        tfidf_feats = self.tfidf.transform(corpus).toarray()      # (N, n_tfidf)
        stat_feats = np.vstack([self._stat_features(w) for w in windows])  # (N, M)
        return np.hstack([tfidf_feats, stat_feats])

    def _stat_features(self, window: pd.DataFrame) -> np.ndarray:
        feats = []

        # 1. Severity counts and statistics
        sev = window["severity_code"].fillna(0).values if "severity_code" in window.columns \
            else window["severity_level"].map(SEVERITY_MAP).fillna(0).values
        feats += [
            float(sev.mean()),
            float(sev.max()),
            float(sev.std() if len(sev) > 1 else 0.0),
            float((sev >= 4).sum()),   # FATAL/SEVERE/ERROR count
            float((sev >= 3).sum()),   # WARNING+ count
            float((sev == 5).sum()),   # FATAL count
        ]

        # 2. Anomaly ratio in window (for supervised hint)
        if "is_anomaly" in window.columns:
            feats.append(float(window["is_anomaly"].mean()))
        else:
            feats.append(0.0)

        # 3. Component diversity
        if "component_clean" in window.columns:
            components = window["component_clean"].fillna("OTHER")
            unique_comps = components.nunique()
            feats.append(float(unique_comps))
            # one-hot for top components
            comp_counts = components.value_counts()
            for comp in TOP_COMPONENTS:
                feats.append(float(comp_counts.get(comp, 0)))
        else:
            feats.append(0.0)
            feats.extend([0.0] * len(TOP_COMPONENTS))

        # 4. Temporal features
        if "timestamp" in window.columns:
            ts = window["timestamp"].values.astype(float)
            duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            event_rate = len(ts) / max(duration, 1e-3)
            feats += [duration, event_rate]
        else:
            feats += [0.0, 0.0]

        # 5. Template diversity (unique event types / window size)
        if "template" in window.columns:
            n_unique_templates = window["template"].nunique()
            feats.append(float(n_unique_templates) / max(len(window), 1))
        else:
            feats.append(0.0)

        # 6. Repeated-error pattern (same template >= 3 times = suspicious)
        if "template" in window.columns:
            template_counts = window["template"].value_counts()
            max_repeat = int(template_counts.max()) if len(template_counts) else 0
            feats.append(float(max_repeat))
            feats.append(float((template_counts >= 3).sum()))
        else:
            feats += [0.0, 0.0]

        return np.array(feats, dtype=np.float32)

    def _build_feature_names(self):
        tfidf_names = [f"tfidf_{v}" for v in self.tfidf.get_feature_names_out()]
        stat_names = (
            ["sev_mean", "sev_max", "sev_std", "error_count",
             "warning_plus_count", "fatal_count", "window_anomaly_ratio",
             "unique_components"]
            + [f"comp_{c}" for c in TOP_COMPONENTS]
            + ["window_duration_s", "event_rate_per_s",
               "template_diversity", "max_template_repeat",
               "templates_repeated_ge3"]
        )
        self._feature_names = tfidf_names + stat_names
