"""
SHAP-based explainability layer for log anomaly detection.
Provides per-prediction and global feature importance attributions.
Reference: Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap library not available - using permutation importance fallback.")


class SHAPExplainer:
    """
    Wraps a trained model to provide SHAP feature attributions.
    Supports TreeExplainer (RF, GBM), LinearExplainer (LR), and KernelExplainer (fallback).
    """

    def __init__(self, model, feature_names: list, model_type: str = "rf",
                 background_samples: int = 100):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.background_samples = background_samples
        self._explainer = None
        self._background = None

    def _get_raw_model(self):
        """Unwrap wrapper class to get sklearn estimator."""
        if hasattr(self.model, "model"):
            return self.model.model
        return self.model

    def fit_background(self, X_background: np.ndarray):
        """Fit explainer on background data (representative sample of training set)."""
        if not SHAP_AVAILABLE:
            return
        n = min(self.background_samples, len(X_background))
        idx = np.random.choice(len(X_background), n, replace=False)
        self._background = X_background[idx]
        raw = self._get_raw_model()
        try:
            if self.model_type in ("rf", "gbm", "tree"):
                self._explainer = shap.TreeExplainer(raw)
                logger.info("SHAP TreeExplainer initialised.")
            elif self.model_type == "lr":
                self._explainer = shap.LinearExplainer(raw, self._background)
                logger.info("SHAP LinearExplainer initialised.")
            else:
                self._explainer = shap.KernelExplainer(
                    lambda x: raw.predict_proba(x)[:, 1],
                    self._background)
                logger.info("SHAP KernelExplainer initialised.")
        except Exception as e:
            logger.warning(f"SHAP explainer init failed: {e} - using permutation fallback.")
            self._explainer = None

    def explain_instance(self, x: np.ndarray) -> dict:
        """
        Return per-feature SHAP values for a single instance.
        x shape: (n_features,)
        """
        x2d = x.reshape(1, -1)
        shap_vals = self._compute_shap(x2d)
        if shap_vals is None:
            return self._permutation_importance_single(x2d)

        vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
        importance = {
            name: float(v)
            for name, v in zip(self.feature_names, vals)
        }
        # return top-10 by absolute value
        top10 = sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        return dict(top10)

    def explain_batch(self, X: np.ndarray) -> list:
        """Return list of per-instance SHAP explanation dicts."""
        return [self.explain_instance(X[i]) for i in range(len(X))]

    def get_global_importance(self, X: np.ndarray) -> dict:
        """Mean absolute SHAP values across a sample - global feature importance."""
        shap_vals = self._compute_shap(X)
        if shap_vals is None:
            return self._permutation_importance_global(X)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = {name: float(v)
                      for name, v in zip(self.feature_names, mean_abs)}
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    def _compute_shap(self, X):
        if not SHAP_AVAILABLE or self._explainer is None:
            return None
        try:
            vals = self._explainer.shap_values(X)
            # Tree/linear explainer returns [neg_class, pos_class] for binary
            if isinstance(vals, list) and len(vals) == 2:
                vals = vals[1]
            return np.array(vals)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return None

    # -- Fallback: permutation-based importance --------------------------------

    def _permutation_importance_single(self, x2d) -> dict:
        """Approximate feature importance via single-instance perturbation."""
        raw = self._get_raw_model()
        try:
            base = raw.predict_proba(x2d)[0, 1]
        except Exception:
            base = float(raw.predict(x2d)[0])

        importances = {}
        for i, name in enumerate(self.feature_names[:20]):
            x_perm = x2d.copy()
            x_perm[0, i] = 0.0
            try:
                perturbed = raw.predict_proba(x_perm)[0, 1]
            except Exception:
                perturbed = float(raw.predict(x_perm)[0])
            importances[name] = float(base - perturbed)

        return dict(sorted(importances.items(),
                            key=lambda kv: abs(kv[1]), reverse=True)[:10])

    def _permutation_importance_global(self, X) -> dict:
        raw = self._get_raw_model()
        n_feat = min(20, X.shape[1])
        try:
            base_scores = raw.predict_proba(X)[:, 1]
        except Exception:
            base_scores = raw.predict(X).astype(float)

        importances = {}
        rng = np.random.default_rng(42)
        for i in range(n_feat):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            try:
                perm_scores = raw.predict_proba(X_perm)[:, 1]
            except Exception:
                perm_scores = raw.predict(X_perm).astype(float)
            importances[self.feature_names[i]] = float(
                np.abs(base_scores - perm_scores).mean())

        return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
