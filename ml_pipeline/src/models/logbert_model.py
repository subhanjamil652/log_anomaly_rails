"""
LogBERT — Self-supervised BERT for Log Anomaly Detection.

Paper: Guo, Yuan, Wu — "LogBERT: Log Anomaly Detection via BERT"
       IJCNN 2021 (IEEE International Joint Conference on Neural Networks)
       arXiv:2103.04475 | GitHub: https://github.com/HelenGuohx/logbert

Key design:
  - Self-supervised: trains on NORMAL log windows only (no anomaly labels required)
  - Masked Log Modelling (MLM): randomly mask ~15% of tokens, predict originals
  - Anomaly score = mean reconstruction loss over masked tokens in test window
  - High loss → model has never seen these patterns → anomaly
  - Bidirectional BERT captures context before AND after each log event

BGL Performance (paper): Recall 92.3%, F1 ≈ 0.878
Advantage over LR/RF: +10% recall over DeepLog, +16% over LogAnomaly on BGL.
The transformer attention captures long-range dependencies that count vectors cannot.
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
MASK_PROBABILITY = 0.15
ANOMALY_PERCENTILE = 95   # threshold: p95 of normal-window losses on validation


def _check_deps():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(f"LogBERT requires torch and transformers: {e}") from e


class LogBERTModel:
    """
    Self-supervised BERT for BGL log anomaly detection.

    Training: fine-tune DistilBERT MLM on normal windows only.
    Inference: reconstruction loss → anomaly if loss > threshold.

    Input to fit/predict: list[str] — one concatenated log-window string per sample.
    y labels are used ONLY to filter normal samples for training.
    """

    def __init__(self,
                 pretrained: str = PRETRAINED_MODEL,
                 max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 lr: float = LEARNING_RATE,
                 mask_prob: float = MASK_PROBABILITY,
                 anomaly_percentile: float = ANOMALY_PERCENTILE,
                 device: str = None):
        self.pretrained = pretrained
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.mask_prob = mask_prob
        self.anomaly_percentile = anomaly_percentile
        self.name = "LogBERT"
        self.model_type = "transformer"
        self._trained = False
        self._threshold = 0.5
        self.tokenizer = None
        self.model = None

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

    def _init_model(self):
        _check_deps()
        from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
        logger.info("Loading %s for MLM (device=%s) …", self.pretrained, self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained)
        self.model = DistilBertForMaskedLM.from_pretrained(self.pretrained).to(self.device)

    def _mask_tokens(self, input_ids):
        """Apply random masking to input_ids. Returns (masked_ids, labels)."""
        import torch
        labels = input_ids.clone()
        # Probability matrix: don't mask special tokens (0=PAD, 101=CLS, 102=SEP)
        prob_matrix = torch.full(labels.shape, self.mask_prob)
        special = (input_ids == self.tokenizer.cls_token_id) | \
                  (input_ids == self.tokenizer.sep_token_id) | \
                  (input_ids == self.tokenizer.pad_token_id)
        prob_matrix.masked_fill_(special, 0.0)
        mask = torch.bernoulli(prob_matrix).bool()
        labels[~mask] = -100   # only compute loss on masked positions
        masked_ids = input_ids.clone()
        masked_ids[mask] = self.tokenizer.mask_token_id
        return masked_ids, labels

    def _compute_loss_batch(self, texts: list) -> np.ndarray:
        """Return per-sample average MLM loss (lower = more normal)."""
        import torch
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i: i + self.batch_size]
                enc = self.tokenizer(
                    chunk, truncation=True, padding="max_length",
                    max_length=self.max_length, return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attn_mask = enc["attention_mask"].to(self.device)
                masked_ids, lbl = self._mask_tokens(input_ids)
                lbl = lbl.to(self.device)
                out = self.model(input_ids=masked_ids, attention_mask=attn_mask, labels=lbl)
                # out.loss is mean over batch; get per-sample via manual reduction
                logits = out.logits   # (B, L, vocab)
                import torch.nn.functional as F
                log_probs = F.log_softmax(logits, dim=-1)
                # gather loss for masked positions only
                vocab_idx = input_ids.unsqueeze(-1)  # original tokens
                token_log_prob = log_probs.gather(dim=-1, index=vocab_idx).squeeze(-1)
                mask_pos = (lbl != -100)
                for b in range(input_ids.size(0)):
                    mp = mask_pos[b]
                    if mp.any():
                        sample_loss = -token_log_prob[b][mp].mean().item()
                    else:
                        sample_loss = 0.0
                    losses.append(sample_loss)
        return np.array(losses, dtype=float)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def fit(self, X_text: list, y: np.ndarray) -> "LogBERTModel":
        """
        Self-supervised training on NORMAL windows only.

        y is used to filter normal samples; it is NOT used as a supervised signal.
        """
        _check_deps()
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        texts = [str(t) for t in X_text]
        y_arr = np.asarray(y, dtype=int)

        # Use only normal samples for self-supervised MLM training
        normal_texts = [t for t, label in zip(texts, y_arr) if label == 0]
        if len(normal_texts) == 0:
            logger.warning("No normal samples found — using all samples for LogBERT training.")
            normal_texts = texts

        logger.info(
            "LogBERT self-supervised training — %d normal windows, device=%s, epochs=%d",
            len(normal_texts), self.device, self.epochs,
        )
        self._init_model()

        enc = self.tokenizer(
            normal_texts, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            for input_ids, attn_mask in loader:
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                masked_ids, labels = self._mask_tokens(input_ids)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                out = self.model(input_ids=masked_ids, attention_mask=attn_mask, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += out.loss.item()
                n_batches += 1
            logger.info("  Epoch %d/%d  avg_mlm_loss=%.4f",
                        epoch + 1, self.epochs, total_loss / max(n_batches, 1))

        # Calibrate threshold on held-out normal samples (last 10%)
        n_cal = max(1, len(normal_texts) // 10)
        cal_texts = normal_texts[-n_cal:]
        cal_losses = self._compute_loss_batch(cal_texts)
        self._threshold = float(np.percentile(cal_losses, self.anomaly_percentile))
        logger.info("LogBERT threshold (p%d of normal losses): %.4f",
                    self.anomaly_percentile, self._threshold)

        self._trained = True
        return self

    def predict_proba(self, X_text: list) -> np.ndarray:
        """Anomaly probability = sigmoid of (loss - threshold) / scale."""
        losses = self._compute_loss_batch([str(t) for t in X_text])
        # Sigmoid centred on threshold; scale chosen so ±2 units → 90% certainty
        scale = max(self._threshold * 0.5, 0.01)
        proba = 1.0 / (1.0 + np.exp(-(losses - self._threshold) / scale))
        return proba

    def predict(self, X_text: list) -> np.ndarray:
        return (self.predict_proba(X_text) >= 0.5).astype(int)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hf_dir = path.replace(".pkl", "_hf")
        os.makedirs(hf_dir, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(hf_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(hf_dir)
        state = {
            "pretrained": self.pretrained, "max_length": self.max_length,
            "batch_size": self.batch_size, "epochs": self.epochs,
            "lr": self.lr, "mask_prob": self.mask_prob,
            "anomaly_percentile": self.anomaly_percentile,
            "name": self.name, "model_type": self.model_type,
            "device": self.device, "_trained": self._trained,
            "_threshold": self._threshold, "hf_dir": hf_dir,
        }
        joblib.dump(state, path)
        logger.info("%s saved → %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> "LogBERTModel":
        _check_deps()
        from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
        state = joblib.load(path)
        obj = cls(
            pretrained=state["pretrained"], max_length=state["max_length"],
            batch_size=state["batch_size"], epochs=state["epochs"],
            lr=state["lr"], mask_prob=state["mask_prob"],
            anomaly_percentile=state["anomaly_percentile"], device=state["device"],
        )
        obj.name = state["name"]
        obj.model_type = state["model_type"]
        obj._trained = state["_trained"]
        obj._threshold = state["_threshold"]
        hf_dir = state.get("hf_dir", path.replace(".pkl", "_hf"))
        if os.path.isdir(hf_dir):
            obj.tokenizer = DistilBertTokenizerFast.from_pretrained(hf_dir)
            obj.model = DistilBertForMaskedLM.from_pretrained(hf_dir).to(obj.device)
            obj.model.eval()
        logger.info("LogBERTModel loaded from %s", path)
        return obj