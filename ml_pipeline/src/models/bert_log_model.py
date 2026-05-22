

"""
BERTLog Model — Fine-tuned DistilBERT for BGL log anomaly detection.

Paradigm (NeuralLog / BERT-Log):
  - Parsing-free: raw log text → WordPiece tokenisation → no Drain needed
  - Each sliding window of log lines → concatenated string → [CLS] representation
  - Binary classification head fine-tuned on BGL window labels
  - Weighted cross-entropy handles ~8% anomaly imbalance

Literature basis:
  BERT-Log  — Zheng et al., Applied AI 2022, DOI:10.1080/08839514.2022.2145642
              F1 = 0.994 on BGL (best supervised result at publication time)
  NeuralLog — Le & Zhang, ASE 2021, arXiv:2108.01955
              F1 = 0.98, eliminates parsing error entirely via WordPiece tokenisation
  Patel 2026 benchmark (arXiv:2604.12218):
              DistilBERT-class models: F1 ≈ 0.961–0.974 vs LR 0.887 / RF 0.912 on BGL
"""

import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

PRETRAINED_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


def _check_deps():
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"BERTLogModel requires torch and transformers: {e}\n"
            "Install: pip install torch transformers"
        ) from e


class BERTLogModel:
    """
    Fine-tuned DistilBERT binary classifier for BGL log window anomaly detection.

    Input to fit/predict/predict_proba: list[str] — one string per window,
    containing the concatenated log-line content for that window.

    No feature engineering required — WordPiece tokenisation handles raw log text
    directly (NeuralLog paradigm: bypasses Drain/Spell parsing error entirely).
    """

    def __init__(self,
                 pretrained: str = PRETRAINED_MODEL,
                 max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 lr: float = LEARNING_RATE,
                 device: str = None):
        self.pretrained = pretrained
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.name = "BERT-Log"
        self.model_type = "transformer"
        self._trained = False
        self.tokenizer = None
        self.model = None
        # Learned on validation data (fit); avoids p(anomaly)≪0.5 miscalibration
        self.decision_threshold = 0.5

        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_model(self):
        _check_deps()
        from transformers import (DistilBertTokenizerFast,
                                   DistilBertForSequenceClassification)
        logger.info("Loading %s tokeniser and model (device=%s) …",
                    self.pretrained, self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.pretrained, num_labels=2
        ).to(self.device)

    def _make_loader(self, texts, labels=None, shuffle=False):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        if labels is not None:
            lbl = torch.tensor(np.asarray(labels, dtype=int), dtype=torch.long)
            ds = TensorDataset(enc["input_ids"], enc["attention_mask"], lbl)
        else:
            ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ------------------------------------------------------------------
    # Public interface (sklearn-compatible, but X is list[str] not ndarray)
    # ------------------------------------------------------------------

    @staticmethod
    def _best_f1_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
        """P(anomaly) threshold that maximises F1 on a labelled set (validation)."""
        from sklearn.metrics import f1_score
        y_true = np.asarray(y_true).ravel()
        p1 = np.asarray(p1, dtype=float).ravel()
        if len(np.unique(y_true)) < 2 or len(p1) == 0:
            return 0.5
        lo, hi = float(p1.min()), float(p1.max())
        if hi <= lo:
            return 0.5
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(lo, hi, 201):
            pred = (p1 >= t).astype(int)
            f1v = f1_score(y_true, pred, zero_division=0)
            if f1v > best_f1:
                best_f1, best_t = f1v, t
        return float(best_t)

    def fit(self, X_text: list, y: np.ndarray,
            X_text_val: list = None, y_val: np.ndarray = None) -> "BERTLogModel":
        """
        Fine-tune DistilBERT.

        Args:
            X_text : list of str — one concatenated log-window string per sample.
            y      : binary int array (0=normal, 1=anomaly).
            X_text_val, y_val: optional validation set for the decision threshold
                (P(anomaly) is often ≪0.5 after fine-tuning on imbalanced logs).
        """
        _check_deps()
        import torch
        import torch.nn as nn
        from transformers import get_linear_schedule_with_warmup

        texts = [str(t) for t in X_text]
        y_arr = np.asarray(y, dtype=int)

        logger.info(
            "Fine-tuning %s — %d windows, device=%s, epochs=%d",
            self.name, len(texts), self.device, self.epochs,
        )
        self._init_model()

        loader = self._make_loader(texts, y_arr, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        total_steps = len(loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps,
        )

        # Weighted loss for class imbalance
        n_neg = int((y_arr == 0).sum())
        n_pos = int((y_arr == 1).sum())
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        weight = torch.tensor([1.0, pos_weight], dtype=torch.float).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            for batch in loader:
                input_ids, attn_mask, labels = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids,
                                    attention_mask=attn_mask).logits
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_batches += 1
            logger.info("  Epoch %d/%d  avg_loss=%.4f",
                        epoch + 1, self.epochs, total_loss / max(n_batches, 1))

        self.model.eval()
        self._trained = True
        if X_text_val is not None and y_val is not None and len(X_text_val) > 0:
            p1_val = self._run_inference([str(t) for t in X_text_val])[:, 1]
            self.decision_threshold = self._best_f1_threshold(
                np.asarray(y_val, dtype=int), p1_val,
            )
            logger.info(
                "%s validation-tuned decision_threshold=%.6f (P(anomaly))",
                self.name, self.decision_threshold,
            )
        else:
            p1_tr = self._run_inference(texts)[:, 1]
            self.decision_threshold = self._best_f1_threshold(y_arr, p1_tr)
            logger.info(
                "%s train-set decision_threshold=%.6f (no validation set passed)",
                self.name, self.decision_threshold,
            )
        logger.info("%s fine-tuning complete.", self.name)
        return self

    def _run_inference(self, X_text: list) -> np.ndarray:
        """Returns probability matrix shape (N, 2)."""
        import torch
        texts = [str(t) for t in X_text]
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i: i + self.batch_size]
                enc = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                out = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def predict(self, X_text: list) -> np.ndarray:
        probs = self._run_inference(X_text)
        p1 = probs[:, 1]
        thr = self.decision_threshold
        if thr is None or (isinstance(thr, float) and np.isnan(thr)):
            thr = 0.5
        return (p1 >= float(thr)).astype(int)

    def predict_proba(self, X_text: list) -> np.ndarray:
        """Returns anomaly probability (class-1 score) for each sample."""
        return self._run_inference(X_text)[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hf_dir = path.replace(".pkl", "_hf")
        os.makedirs(hf_dir, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(hf_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(hf_dir)
        state = {
            "pretrained": self.pretrained,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "name": self.name,
            "model_type": self.model_type,
            "device": self.device,
            "_trained": self._trained,
            "hf_dir": hf_dir,
            "decision_threshold": getattr(self, "decision_threshold", 0.5),
        }
        joblib.dump(state, path)
        logger.info("%s saved → %s  (HF weights → %s)", self.name, path, hf_dir)

    @classmethod
    def load(cls, path: str) -> "BERTLogModel":
        _check_deps()
        from transformers import (DistilBertTokenizerFast,
                                   DistilBertForSequenceClassification)
        state = joblib.load(path)
        obj = cls(
            pretrained=state["pretrained"],
            max_length=state["max_length"],
            batch_size=state["batch_size"],
            epochs=state["epochs"],
            lr=state["lr"],
            device=state["device"],
        )
        obj.name = state["name"]
        obj.model_type = state["model_type"]
        obj._trained = state["_trained"]
        dt = state.get("decision_threshold", None)
        if dt is None or (isinstance(dt, (int, float)) and abs(float(dt) - 0.5) < 1e-9):
            # Default 0.5 is wrong for typical BGL fine-tunes (p(anomaly) is O(10^-2)).
            obj.decision_threshold = 0.0195
            logger.info(
                "%s: using calibrated P(anomaly) threshold %.4f (legacy or default 0.5)",
                obj.name, obj.decision_threshold,
            )
        else:
            obj.decision_threshold = float(dt)
        hf_dir = state.get("hf_dir", path.replace(".pkl", "_hf"))
        if os.path.isdir(hf_dir):
            obj.tokenizer = DistilBertTokenizerFast.from_pretrained(hf_dir)
            obj.model = DistilBertForSequenceClassification.from_pretrained(
                hf_dir
            ).to(obj.device)
            obj.model.eval()
        logger.info("BERTLogModel loaded from %s", path)
        return obj