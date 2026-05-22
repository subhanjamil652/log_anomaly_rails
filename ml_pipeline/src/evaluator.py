"""
Model Evaluation - Precision, Recall, F1, AUC-ROC, FPR, FNR, latency.
"""

import time
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve,
)

logger = logging.getLogger(__name__)


def _as_float(x) -> float:
    """Coerce sklearn/numpy metric outputs to a plain Python float for round()."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    if arr.size != 1:
        return float(arr[-1])
    return float(arr[0])


class ModelEvaluator:
    """Evaluate and compare all candidate anomaly detection models."""

    def evaluate(self, model, X_test: np.ndarray,
                 y_test: np.ndarray, model_name: str = None) -> dict:
        name = model_name or getattr(model, "name", "Model")
        logger.info(f"Evaluating {name} ...")

        y_test = np.asarray(y_test).ravel()
        # Latency measurement
        start = time.perf_counter()
        y_pred = np.asarray(model.predict(X_test)).ravel()
        latency_ms = (time.perf_counter() - start) / len(X_test) * 1000

        y_proba = None
        y_score = None
        try:
            y_proba = model.predict_proba(X_test)
            raw = np.asarray(y_proba, dtype=float)
            if raw.ndim == 2 and raw.shape[1] >= 2:
                y_score = raw[:, 1]
            else:
                y_score = raw.ravel()
        except Exception:
            pass

        # Report **classification** metrics from `model.predict` only (same as /predict).
        # Tuning the threshold *on this same holdout* to maximise F1 inflates F1/ACC
        # toward 1.0 on small test sets and is not credible for a thesis.
        # BERT-Log encodes a calibrated `decision_threshold` in the checkpoint; AUC
        # still uses the continuous score (ranking), which is valid.
        decision_t = float(getattr(model, "decision_threshold", 0.5) or 0.5)

        prec = _as_float(precision_score(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0,
        ))
        rec = _as_float(recall_score(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0,
        ))
        f1 = _as_float(f1_score(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0,
        ))
        acc = _as_float(accuracy_score(y_test, y_pred))
        if y_score is not None:
            try:
                auc = _as_float(roc_auc_score(y_test, y_score))
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        roc_data = None
        if y_score is not None:
            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_score)
            roc_data = {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()}

        n_eval = int(len(y_test))
        result = {
            "model_name": name,
            "model_type": getattr(model, "model_type", "unknown"),
            "decision_threshold": round(float(decision_t), 6),
            "n_eval_samples": n_eval,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_score":  round(f1, 4),
            "auc_roc":   round(auc, 4),
            "accuracy":  round(acc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "detection_latency_ms": round(latency_ms, 4),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "roc_curve": roc_data,
        }
        if n_eval < 200:
            result["metric_note"] = (
                f"Small holdout (n={n_eval} test windows). Scores of 1.000 on accuracy/F1 here "
                f"only mean the model made no (or no harmful) errors on this slice — not proof of perfect "
                f"real-world BGL coverage. Cite TP/FP/TN/FN (below) in your report; for stronger claims, "
                f"increase the held-out n or add cross-validation. AUC-ROC is usually more stable than a single "
                f"binary accuracy on tiny n."
            )
        elif acc >= 0.999 and n_eval < 2000:
            result["metric_note"] = (
                f"Very high accuracy ({acc:.3f}) on n={n_eval} — still a single split; report confusion counts "
                "and, where possible, validate on a larger or temporal holdout."
            )
        logger.info(
            f"  {name}: Precision={prec:.4f}  Recall={rec:.4f}  "
            f"F1={f1:.4f}  AUC-ROC={auc:.4f}  Latency={latency_ms:.3f}ms/sample"
        )
        return result

    def compare_models(self, results: dict) -> pd.DataFrame:
        rows = []
        for name, r in results.items():
            rows.append({
                "Model": r["model_name"],
                "Type": r["model_type"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1_score"],
                "AUC-ROC": r["auc_roc"],
                "Accuracy": r["accuracy"],
                "FPR": r["false_positive_rate"],
                "FNR": r["false_negative_rate"],
                "Latency (ms/sample)": r["detection_latency_ms"],
            })
        df = pd.DataFrame(rows).sort_values("F1", ascending=False)
        return df

    def generate_report(self, results: dict) -> str:
        df = self.compare_models(results)
        lines = [
            "=" * 70,
            "BGL LOG ANOMALY DETECTION - MODEL EVALUATION REPORT",
            "=" * 70,
            "",
            df.to_string(index=False),
            "",
            "-" * 70,
            f"Best model by F1-score: {df.iloc[0]['Model']}",
            f"  F1={df.iloc[0]['F1']:.4f}  "
            f"Precision={df.iloc[0]['Precision']:.4f}  "
            f"Recall={df.iloc[0]['Recall']:.4f}  "
            f"AUC-ROC={df.iloc[0]['AUC-ROC']:.4f}",
            "-" * 70,
        ]
        return "\n".join(lines)
