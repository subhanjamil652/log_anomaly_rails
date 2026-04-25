"""
Training Orchestrator - runs the full BGL log anomaly detection pipeline.

Flow:
  1. Load BGL data (file or BGL-format proxy when missing)
  2. Parse with Drain
  3. Feature engineering + sliding windows
  4. SMOTE oversampling
  5. Train all 4 models
  6. Evaluate + select best
  7. If best F1 < F1_THRESHOLD -> apply optimised fallback model
  8. Save artifacts
"""

import os
import sys
import json
import time
import logging
import subprocess
import numpy as np
import joblib
from datetime import datetime

from .data_loader import BGLDataLoader
from .drain_parser import DrainParser
from .feature_engineering import FeatureEngineer
from .models.logistic_regression_model import LogisticRegressionModel
from .models.random_forest_model import RandomForestModel
from .models.isolation_forest_model import IsolationForestModel
from .models.lstm_autoencoder import LSTMAutoencoder
from .models.bert_log_model import BERTLogModel
from .models.logbert_model import LogBERTModel
from .models.plelog_model import PLELogModel
from .models.logformer_model import LogFormerModel
from .models.loggpt_model import LogGPTModel
from .evaluator import ModelEvaluator
from .shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

F1_THRESHOLD = 0.88          # Minimum acceptable F1 before fallback
WINDOW_SIZE  = 20
STRIDE       = 10
# Larger test holdout + chronological split → realistic metrics (no overlapping-window leakage)
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15


class AnomalyDetectionTrainer:
    """Full training pipeline for BGL log anomaly detection."""

    def __init__(self, save_dir: str = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = save_dir or os.path.join(base, "saved_models")
        self._training_data_file = None
        self._raw_windows = None   # list[pd.DataFrame] kept for BERT text input
        os.makedirs(self.save_dir, exist_ok=True)

    def _base_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve_data_path(self, data_path: str | None, force_proxy_bgl: bool = False) -> tuple[str | None, bool]:
        """
        Returns (path_or_none, use_proxy_bgl).
        Prefers explicit path, then ml_pipeline/data/BGL.log, then download script.
        """
        if force_proxy_bgl:
            return None, True

        base = self._base_dir()
        candidates = []
        if data_path:
            candidates.append(os.path.abspath(data_path))
        candidates.append(os.path.join(base, "data", "BGL.log"))
        candidates.append(os.path.join(base, "data", "BGL_2k.log"))

        for p in candidates:
            if p and os.path.isfile(p):
                logger.info("Using real BGL logs: %s", p)
                return p, False

        dl = self._try_download_bgl()
        if dl and os.path.isfile(dl):
            logger.info("Using downloaded BGL logs: %s", dl)
            return dl, False

        logger.warning(
            "No BGL.log found — training on a BGL-format proxy dataset. "
            "For production-like metrics, run: python scripts/download_bgl.py "
            "then re-run training."
        )
        return None, True

    def _try_download_bgl(self) -> str | None:
        script = os.path.join(self._base_dir(), "scripts", "download_bgl.py")
        if not os.path.isfile(script):
            return None
        logger.info("Attempting to download BGL dataset from LogHub (Zenodo) …")
        try:
            subprocess.run(
                [sys.executable, script],
                cwd=self._base_dir(),
                timeout=900,
                check=False,
            )
        except Exception as e:
            logger.warning("download_bgl.py: %s", e)
        out = os.path.join(self._base_dir(), "data", "BGL.log")
        return out if os.path.isfile(out) else None

    def _chronological_split(self, X: np.ndarray, y: np.ndarray):
        """
        Time-ordered split (windows follow log order). Reduces train/test leakage from
        overlapping sliding windows vs i.i.d. stratified split (which inflates scores).
        """
        n = X.shape[0]
        test_n = max(1, int(round(n * TEST_SIZE)))
        test_n = min(test_n, max(1, n - 20))
        trainval_n = n - test_n
        val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
        val_n = int(round(trainval_n * val_ratio))
        val_n = max(1, min(val_n, trainval_n - 10))
        t_end = trainval_n - val_n
        X_train = X[:t_end]
        y_train = y[:t_end]
        X_val = X[t_end:trainval_n]
        y_val = y[t_end:trainval_n]
        X_test = X[trainval_n:]
        y_test = y[trainval_n:]
        return X_train, X_val, X_test, y_train, y_val, y_test

    # -- Main entry point ------------------------------------------------------

    def run_full_pipeline(self, data_path: str = None,
                          proxy_samples: int = 80_000,
                          force_proxy_bgl: bool = False) -> dict:
        """
        Execute the complete training pipeline.
        If data_path is None or file not found, a BGL-format proxy corpus is generated.
        """
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("BGL Log Anomaly Detection - Training Pipeline")
        logger.info("=" * 60)

        # -- 1. Data Loading --------------------------------------------------
        parser = DrainParser(depth=4, sim_threshold=0.4)
        loader = BGLDataLoader(drain_parser=parser)

        resolved, use_proxy_bgl = self._resolve_data_path(data_path, force_proxy_bgl=force_proxy_bgl)
        self._training_data_file = resolved

        if not use_proxy_bgl and resolved:
            logger.info("Loading real BGL dataset (sample up to 400k lines) …")
            df = loader.load_sample(resolved, n=400_000)
        else:
            logger.info(
                "Generating BGL-format proxy logs — prefer data/BGL.log for "
                "production-like evaluation metrics."
            )
            df = loader.generate_bgl_proxy(n_samples=proxy_samples)

        logger.info(f"Dataset: {len(df):,} entries, "
                    f"{df['is_anomaly'].sum():,} anomalies "
                    f"({df['is_anomaly'].mean()*100:.1f}%)")

        # -- 2. Sliding windows -----------------------------------------------
        windows, y = loader.create_windows(df, window_size=WINDOW_SIZE, stride=STRIDE)
        self._raw_windows = windows   # retained for BERT text-based training
        logger.info(f"Windows created: {len(windows):,} | "
                    f"Anomalous: {y.sum():,} ({y.mean()*100:.1f}%)")

        # -- 3. Feature Engineering -------------------------------------------
        fe = FeatureEngineer(window_size=WINDOW_SIZE, n_tfidf_features=50)
        X, y = fe.fit_transform(windows, y)
        feature_names = fe.get_feature_names()
        fe.save(os.path.join(self.save_dir, "feature_engineer.pkl"))
        logger.info(f"Feature matrix: {X.shape}")

        # -- 4. Train/Val/Test split (chronological; no stratify — avoids leakage) ---
        X_train, X_val, X_test, y_train, y_val, y_test = self._chronological_split(X, y)
        logger.info(
            "Chronological split — Train: %s  Val: %s  Test: %s "
            "(anomaly %% train/val/test: %.1f / %.1f / %.1f)",
            f"{len(X_train):,}",
            f"{len(X_val):,}",
            f"{len(X_test):,}",
            100.0 * y_train.mean(),
            100.0 * y_val.mean(),
            100.0 * y_test.mean(),
        )

        # Persist holdout for API runtime metrics (same split as training eval)
        holdout_npz = os.path.join(self.save_dir, "eval_holdout.npz")
        np.savez_compressed(holdout_npz, X_test=X_test, y_test=y_test)
        logger.info(f"Saved holdout evaluation set: {holdout_npz}")

        # -- 5. SMOTE (feature-based models only) ------------------------------
        X_train_res, y_train_res = self._apply_smote(X_train, y_train)

        # -- 5b. Chronological window-text splits for BERT --------------------
        n_win = len(windows)
        n_test_w  = max(1, int(round(n_win * TEST_SIZE)))
        n_test_w  = min(n_test_w, max(1, n_win - 20))
        n_tv_w    = n_win - n_test_w
        val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
        n_val_w   = max(1, min(int(round(n_tv_w * val_ratio)), n_tv_w - 10))
        t_end_w   = n_tv_w - n_val_w
        win_train = windows[:t_end_w]
        win_val   = windows[t_end_w:n_tv_w]
        win_test  = windows[n_tv_w:]
        X_text_train = self._windows_to_text(win_train)
        X_text_val   = self._windows_to_text(win_val)
        X_text_test  = self._windows_to_text(win_test)
        y_text_train = y[:t_end_w]
        y_text_val   = y[t_end_w:n_tv_w]
        y_text_test  = y[n_tv_w:]

        # Persist text-based holdout for BERT-Log API evaluation
        text_holdout_npz = os.path.join(self.save_dir, "eval_holdout_text.npz")
        np.savez_compressed(
            text_holdout_npz,
            X_text_test=np.array(X_text_test, dtype=object),
            y_text_test=y_text_test,
        )
        logger.info(f"Saved BERT text holdout: {text_holdout_npz}")

        # -- 6. Train all models -----------------------------------------------
        results = self.train_all_models(
            X_train_res, X_val, X_test,
            y_train_res, y_val, y_test,
            feature_names=feature_names,
            X_text_train=X_text_train,
            X_text_val=X_text_val,
            X_text_test=X_text_test,
            y_text_train=y_text_train,
            y_text_val=y_text_val,
            y_text_test=y_text_test,
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
            "data_source": (
                "bgl_proxy"
                if use_proxy_bgl
                else os.path.basename(self._training_data_file or "BGL.log")
            ),
            "eval_split": "chronological",
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
                         feature_names=None,
                         X_text_train=None, X_text_val=None, X_text_test=None,
                         y_text_train=None, y_text_val=None, y_text_test=None) -> dict:
        evaluator = ModelEvaluator()
        results = {}
        feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # ---- Feature-based models (sklearn-compatible, work on X numpy array) ----
        feature_models = [
            ("Logistic Regression", LogisticRegressionModel()),
            ("Random Forest",       RandomForestModel(n_estimators=200)),
            ("Isolation Forest",    IsolationForestModel(contamination=0.08)),
            ("LSTM Autoencoder",    LSTMAutoencoder(input_size=1)),
        ]

        for name, model in feature_models:
            logger.info(f"\n-- Training: {name} --")
            try:
                model.fit(X_train, y_train)
                result = evaluator.evaluate(model, X_test, y_test, name)
                results[name] = result
                self._save_model(model, self._safe_fname(name))

                # SHAP is optional — must not wipe a good train/eval result on failure
                if name in ("Random Forest", "Logistic Regression"):
                    try:
                        mtype = "rf" if "Forest" in name else "lr"
                        explainer = SHAPExplainer(model, feature_names, model_type=mtype)
                        explainer.fit_background(X_train[:200])
                        global_imp = explainer.get_global_importance(X_test[:100])
                        results[name]["shap_global"] = {k: v for k, v in
                                                        list(global_imp.items())[:15]}
                        joblib.dump(explainer,
                                    os.path.join(self.save_dir,
                                                 f"shap_{self._safe_fname(name)}.pkl"))
                    except Exception as shap_err:
                        logger.warning("SHAP skipped for %s: %s", name, shap_err)

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

        # ---- Text-based transformer/deep models (NeuralLog / BERT-Log paradigm) ----
        if X_text_train is not None and X_text_test is not None:
            text_models = [
                # (display_name, model_instance, save_filename)
                ("BERT-Log",   BERTLogModel(),   "bert_log"),
                ("LogBERT",    LogBERTModel(),   "logbert"),
                ("PLELog",     PLELogModel(),    "plelog"),
                ("LogFormer",  LogFormerModel(), "logformer"),
                ("LogGPT",     LogGPTModel(),    "loggpt"),
            ]
            for name, model, fname in text_models:
                logger.info("\n-- Training: %s --", name)
                try:
                    model.fit(X_text_train, y_text_train)
                    result = evaluator.evaluate(model, X_text_test, y_text_test, model.name)
                    results[model.name] = result
                    self._save_model(model, fname)
                except Exception as e:
                    logger.error("Failed to train %s: %s", name, e)
                    results[name] = {
                        "model_name": name, "f1_score": 0.0,
                        "precision": 0.0, "recall": 0.0,
                        "auc_roc": 0.5, "accuracy": 0.0,
                        "false_positive_rate": 1.0, "false_negative_rate": 1.0,
                        "detection_latency_ms": 0.0,
                        "model_type": "transformer", "error": str(e),
                    }

        return results

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _windows_to_text(windows: list) -> list:
        """
        Convert a list of window DataFrames to a list of strings for BERT input.

        Concatenates the 'content' field of every log line in the window,
        separated by ' [SEP] '. No Drain parsing needed — BERT's WordPiece
        tokeniser handles raw log text directly (NeuralLog paradigm).
        """
        texts = []
        for win_df in windows:
            if hasattr(win_df, "iterrows"):
                parts = win_df["content"].astype(str).tolist()
            else:
                parts = [str(win_df)]
            texts.append(" [SEP] ".join(parts))
        return texts

    def _apply_smote(self, X, y):
        try:
            from imblearn.over_sampling import SMOTE
            logger.info("Applying SMOTE oversampling ...")
            sm = SMOTE(sampling_strategy=0.5, random_state=42)
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
