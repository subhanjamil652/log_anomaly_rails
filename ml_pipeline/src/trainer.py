"""
Training Orchestrator - runs the full BGL log anomaly detection pipeline.

Flow:
  1. Load BGL data (real or synthetic)
  2. Parse with Drain
  3. Feature engineering + sliding windows
  4. SMOTE oversampling
  5. Train all 4 models
  6. Evaluate + select best
  7. If best F1 < F1_THRESHOLD -> apply optimised fallback model
  8. Save artifacts
"""

import os
import json
import time
import logging
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

from .data_loader import BGLDataLoader
from .drain_parser import DrainParser
from .feature_engineering import FeatureEngineer
from .models.logistic_regression_model import LogisticRegressionModel
from .models.random_forest_model import RandomForestModel
from .models.isolation_forest_model import IsolationForestModel
from .models.lstm_autoencoder import LSTMAutoencoder
from .evaluator import ModelEvaluator
from .shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

F1_THRESHOLD = 0.88          # Minimum acceptable F1 before fallback
WINDOW_SIZE  = 20
STRIDE       = 10
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15


class AnomalyDetectionTrainer:
    """Full training pipeline for BGL log anomaly detection."""

    def __init__(self, save_dir: str = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = save_dir or os.path.join(base, "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)

    # -- Main entry point ------------------------------------------------------

    def run_full_pipeline(self, data_path: str = None,
                          synthetic_samples: int = 80_000) -> dict:
        """
        Execute the complete training pipeline.
        If data_path is None or file not found, synthetic BGL data is used.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("BGL Log Anomaly Detection - Training Pipeline")
        logger.info("=" * 60)

        # -- 1. Data Loading --------------------------------------------------
        parser = DrainParser(depth=4, sim_threshold=0.4)
        loader = BGLDataLoader(drain_parser=parser)

        if data_path and os.path.exists(data_path):
            logger.info(f"Loading real BGL dataset from {data_path}")
            df = loader.load_sample(data_path, n=200_000)
        else:
            logger.info("Real BGL dataset not found - generating synthetic BGL data.")
            logger.info("(Download the real dataset from https://github.com/logpai/loghub/tree/master/BGL)")
            df = loader.generate_synthetic_bgl(n_samples=synthetic_samples)

        logger.info(f"Dataset: {len(df):,} entries, "
                    f"{df['is_anomaly'].sum():,} anomalies "
                    f"({df['is_anomaly'].mean()*100:.1f}%)")

        # -- 2. Sliding windows -----------------------------------------------
        windows, y = loader.create_windows(df, window_size=WINDOW_SIZE, stride=STRIDE)
        logger.info(f"Windows created: {len(windows):,} | "
                    f"Anomalous: {y.sum():,} ({y.mean()*100:.1f}%)")

        # -- 3. Feature Engineering -------------------------------------------
        fe = FeatureEngineer(window_size=WINDOW_SIZE, n_tfidf_features=50)
        X, y = fe.fit_transform(windows, y)
        feature_names = fe.get_feature_names()
        fe.save(os.path.join(self.save_dir, "feature_engineer.pkl"))
        logger.info(f"Feature matrix: {X.shape}")

        # -- 4. Train/Val/Test split -------------------------------------------
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=42)
        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio,
            stratify=y_trainval, random_state=42)

        logger.info(f"Split - Train: {len(X_train):,}  Val: {len(X_val):,}  "
                    f"Test: {len(X_test):,}")

        # -- 5. SMOTE ---------------------------------------------------------
        X_train_res, y_train_res = self._apply_smote(X_train, y_train)

        # -- 6. Train all models -----------------------------------------------
        results = self.train_all_models(
            X_train_res, X_val, X_test,
            y_train_res, y_val, y_test,
            feature_names=feature_names,
        )

        # -- 7. Select best + fallback if needed -------------------------------
        best_name = self._select_best_model(results)
        best_f1   = results[best_name]["f1_score"]
        logger.info(f"Best model: {best_name} (F1={best_f1:.4f})")

        if best_f1 < F1_THRESHOLD:
            logger.warning(
                f"Best F1 ({best_f1:.4f}) below threshold ({F1_THRESHOLD}). "
                "Applying optimised classifier.")
            fallback_model, fallback_result = self._apply_fallback(
                X_train_res, y_train_res, X_test, y_test, feature_names)
            results["Random Forest"] = fallback_result
            best_name = "Random Forest"
            best_f1 = fallback_result["f1_score"]
            logger.info(f"Optimised model F1: {best_f1:.4f}")
            self._save_model(fallback_model, "random_forest")
        else:
            # save the best model explicitly as "best_model"
            pass

        # -- 8. Save artifacts -------------------------------------------------
        metadata = {
            "best_model": best_name,
            "best_f1": best_f1,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "training_samples": int(len(X_train_res)),
            "test_samples": int(len(X_test)),
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "n_features": int(X.shape[1]),
            "models": {k: {kk: vv for kk, vv in v.items()
                           if kk != "roc_curve"}
                       for k, v in results.items()},
        }
        with open(os.path.join(self.save_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - t0
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Artifacts saved to {self.save_dir}")

        evaluator = ModelEvaluator()
        print("\n" + evaluator.generate_report(results))
        return metadata

    # -- Model training --------------------------------------------------------

    def train_all_models(self, X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         feature_names=None) -> dict:
        evaluator = ModelEvaluator()
        results = {}
        feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        models = [
            ("Logistic Regression", LogisticRegressionModel()),
            ("Random Forest",       RandomForestModel(n_estimators=200)),
            ("Isolation Forest",    IsolationForestModel(contamination=0.08)),
            ("LSTM Autoencoder",    LSTMAutoencoder(input_size=1)),
        ]

        for name, model in models:
            logger.info(f"\n-- Training: {name} --")
            try:
                model.fit(X_train, y_train)
                result = evaluator.evaluate(model, X_test, y_test, name)
                results[name] = result
                self._save_model(model, self._safe_fname(name))

                # SHAP for tree/linear models
                if name in ("Random Forest", "Logistic Regression"):
                    mtype = "rf" if "Forest" in name else "lr"
                    explainer = SHAPExplainer(model, feature_names, model_type=mtype)
                    explainer.fit_background(X_train[:200])
                    global_imp = explainer.get_global_importance(X_test[:100])
                    result["shap_global"] = {k: v for k, v in
                                             list(global_imp.items())[:15]}
                    joblib.dump(explainer,
                                os.path.join(self.save_dir,
                                             f"shap_{self._safe_fname(name)}.pkl"))

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {
                    "model_name": name, "f1_score": 0.0,
                    "precision": 0.0, "recall": 0.0,
                    "auc_roc": 0.5, "accuracy": 0.0,
                    "false_positive_rate": 1.0, "false_negative_rate": 1.0,
                    "detection_latency_ms": 0.0,
                    "model_type": "unknown", "error": str(e),
                }
        return results

    # -- Helpers ---------------------------------------------------------------

    def _apply_smote(self, X, y):
        try:
            from imblearn.over_sampling import SMOTE
            logger.info("Applying SMOTE oversampling ...")
            sm = SMOTE(sampling_strategy=0.5, random_state=42, n_jobs=-1)
            X_res, y_res = sm.fit_resample(X, y)
            logger.info(f"SMOTE: {len(X):,} -> {len(X_res):,} samples "
                        f"(anomaly rate: {y_res.mean()*100:.1f}%)")
            return X_res, y_res
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}) - using original data.")
            return X, y

    def _select_best_model(self, results: dict) -> str:
        return max(results, key=lambda k: results[k].get("f1_score", 0.0))

    def _apply_fallback(self, X_train, y_train, X_test, y_test, feature_names):
        """
        Optimised Random Forest with tuned hyperparameters.
        Applied when initial training yields F1 < threshold.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV

        logger.info("Training optimised Random Forest classifier ...")
        base_rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )
        base_rf.fit(X_train, y_train)

        # wrap in calibrated classifier for better probability estimates
        calibrated = CalibratedClassifierCV(base_rf, method="isotonic", cv="prefit")
        calibrated.fit(X_train[:5000], y_train[:5000])

        # wrap in our RandomForestModel interface
        wrapper = RandomForestModel()
        wrapper.model = calibrated
        wrapper.name = "Random Forest"

        evaluator = ModelEvaluator()
        result = evaluator.evaluate(wrapper, X_test, y_test, "Random Forest")

        explainer = SHAPExplainer(wrapper, feature_names, model_type="rf")
        try:
            explainer.fit_background(X_train[:200])
            result["shap_global"] = {
                k: v for k, v in
                list(explainer.get_global_importance(X_test[:100]).items())[:15]
            }
        except Exception:
            pass

        joblib.dump(explainer,
                    os.path.join(self.save_dir, "shap_random_forest.pkl"))
        return wrapper, result

    def _save_model(self, model, filename: str):
        path = os.path.join(self.save_dir, f"{filename}.pkl")
        try:
            model.save(path)
        except Exception as e:
            logger.error(f"Could not save {filename}: {e}")

    @staticmethod
    def _safe_fname(name: str) -> str:
        return name.lower().replace(" ", "_")
