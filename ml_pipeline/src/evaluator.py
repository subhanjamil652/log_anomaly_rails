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


class ModelEvaluator:
    """Evaluate and compare all candidate anomaly detection models."""

    def evaluate(self, model, X_test: np.ndarray,
                 y_test: np.ndarray, model_name: str = None) -> dict:
        name = model_name or getattr(model, "name", "Model")
        logger.info(f"Evaluating {name} ...")

        # Latency measurement
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        latency_ms = (time.perf_counter() - start) / len(X_test) * 1000

        y_proba = None
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.5

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        roc_data = None
        if y_proba is not None:
            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
            roc_data = {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()}

        result = {
            "model_name": name,
            "model_type": getattr(model, "model_type", "unknown"),
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
