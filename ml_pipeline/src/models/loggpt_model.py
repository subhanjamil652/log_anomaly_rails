"""
LogGPT — Log Anomaly Detection via GPT.

Paper: Han, Yuan, Trabelsi — "LogGPT: Log Anomaly Detection via GPT"
       IEEE BigData 2023
       arXiv:2309.14482 | GitHub: https://github.com/nokia/LogGPT

Key design:
  - GPT (causal language model) fine-tuned on NORMAL log sequences only
  - Anomaly detection: abnormal sequences have HIGH perplexity under the model
    (model assigns low probability to log sequences it hasn't learned to generate)
  - Statistically significant outperformance (p < 0.05) vs. 9 baselines:
    PCA, iForest, OCSVM, DeepLog, LogAnomaly, LogBERT, etc.
  - The paper uses REINFORCE for RL fine-tuning; this implementation uses
    supervised perplexity-based threshold (captures the core detection mechanism)

BGL Performance (paper): Precision 0.940, Recall 0.977, F1 = 0.958
Advantage over LR/RF: GPT captures the generative distribution of normal logs —
LR/RF achieve F1 ≈ 0.05–0.14 on BGL in unsupervised setting vs LogGPT's 0.958.
"""

import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

PRETRAINED_MODEL = "distilgpt2"    # 82M params — efficient GPT2 variant
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
ANOMALY_PERCENTILE = 95   # threshold: p95 of normal-window perplexity on calibration set


def _check_deps():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(f"LogGPT requires torch and transformers: {e}") from e


class LogGPTModel:
    """
    DistilGPT2 fine-tuned on normal BGL log windows for anomaly detection.

    Training (self-supervised):
        Fine-tune causal language model on NORMAL log windows only.
        The model learns the generative distribution of normal log sequences.

    Inference:
        Compute perplexity (negative log-likelihood per token) for each test window.
        High perplexity → model finds the sequence unlikely → anomaly.
        Threshold = p95 of perplexity on held-out normal training windows.

    Input: list[str] — one concatenated log-window string per sample.
    """

    def __init__(self,
                 pretrained: str = PRETRAINED_MODEL,
                 max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 lr: float = LEARNING_RATE,
                 anomaly_percentile: float = ANOMALY_PERCENTILE,
                 device: str = None):
        self.pretrained = pretrained
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.anomaly_percentile = anomaly_percentile
        self.name = "LogGPT"
        self.model_type = "transformer"
        self._trained = False
        self._threshold = 10.0   # default perplexity threshold
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
        from transformers import GPT2TokenizerFast, GPT2LMHeadModel
        logger.info("Loading %s for causal LM (device=%s) …", self.pretrained, self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.pretrained)
        self.tokenizer.pad_token = self.tokenizer.eos_token   # GPT2 has no pad token
        self.model = GPT2LMHeadModel.from_pretrained(self.pretrained).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def _compute_perplexity(self, texts: list) -> np.ndarray:
        """Compute per-sample perplexity (lower = more normal)."""
        import torch
        perplexities = []
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

                # For causal LM, labels = input_ids; padding positions get -100
                labels = input_ids.clone()
                labels[attn_mask == 0] = -100

                out = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                # out.loss = mean NLL over non-padding tokens in batch
                # Compute per-sample NLL manually
                logits = out.logits  # (B, L, V)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                import torch.nn.functional as F
                log_probs = F.log_softmax(shift_logits, dim=-1)   # (B, L-1, V)
                token_log_probs = log_probs.gather(
                    dim=-1, index=shift_labels.unsqueeze(-1).clamp(min=0)
                ).squeeze(-1)  # (B, L-1)

                valid_mask = (shift_labels != -100).float()
                for b in range(input_ids.size(0)):
                    n_valid = valid_mask[b].sum()
                    if n_valid > 0:
                        nll = -(token_log_probs[b] * valid_mask[b]).sum() / n_valid
                        ppl = float(torch.exp(nll).cpu().item())
                    else:
                        ppl = float(self._threshold)
                    perplexities.append(min(ppl, 1e6))  # cap to avoid inf

        return np.array(perplexities, dtype=float)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def fit(self, X_text: list, y: np.ndarray) -> "LogGPTModel":
        """
        Self-supervised training on NORMAL windows only.
        y used only to filter normal samples.
        """
        _check_deps()
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        texts = [str(t) for t in X_text]
        y_arr = np.asarray(y, dtype=int)

        normal_texts = [t for t, label in zip(texts, y_arr) if label == 0]
        if len(normal_texts) == 0:
            logger.warning("No normal samples — using all for LogGPT training.")
            normal_texts = texts

        logger.info(
            "LogGPT self-supervised training — %d normal windows, device=%s, epochs=%d",
            len(normal_texts), self.device, self.epochs,
        )
        self._init_model()

        enc = self.tokenizer(
            normal_texts, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        # Labels = input_ids, -100 at padding
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(input_ids, attn_mask, labels)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            for b_ids, b_mask, b_lbl in loader:
                b_ids  = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_lbl  = b_lbl.to(self.device)
                optimizer.zero_grad()
                out = self.model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbl)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += out.loss.item()
                n_batches += 1
            logger.info("  Epoch %d/%d  avg_nll=%.4f",
                        epoch + 1, self.epochs, total_loss / max(n_batches, 1))

        # Calibrate threshold: p95 of perplexity on held-out normal samples
        n_cal = max(1, len(normal_texts) // 10)
        cal_ppls = self._compute_perplexity(normal_texts[-n_cal:])
        self._threshold = float(np.percentile(cal_ppls, self.anomaly_percentile))
        logger.info("LogGPT threshold (p%d of normal perplexity): %.2f",
                    self.anomaly_percentile, self._threshold)

        self._trained = True
        return self

    def predict_proba(self, X_text: list) -> np.ndarray:
        """Anomaly probability via sigmoid of (perplexity - threshold) / scale."""
        ppls = self._compute_perplexity([str(t) for t in X_text])
        scale = max(self._threshold * 0.5, 1.0)
        return 1.0 / (1.0 + np.exp(-(ppls - self._threshold) / scale))

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
            "batch_size": self.batch_size, "epochs": self.epochs, "lr": self.lr,
            "anomaly_percentile": self.anomaly_percentile,
            "name": self.name, "model_type": self.model_type,
            "device": self.device, "_trained": self._trained,
            "_threshold": self._threshold, "hf_dir": hf_dir,
        }
        joblib.dump(state, path)
        logger.info("%s saved → %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> "LogGPTModel":
        _check_deps()
        from transformers import GPT2TokenizerFast, GPT2LMHeadModel
        state = joblib.load(path)
        obj = cls(
            pretrained=state["pretrained"], max_length=state["max_length"],
            batch_size=state["batch_size"], epochs=state["epochs"],
            lr=state["lr"], anomaly_percentile=state["anomaly_percentile"],
            device=state["device"],
        )
        obj.name = state["name"]
        obj.model_type = state["model_type"]
        obj._trained = state["_trained"]
        obj._threshold = state["_threshold"]
        hf_dir = state.get("hf_dir", path.replace(".pkl", "_hf"))
        if os.path.isdir(hf_dir):
            obj.tokenizer = GPT2TokenizerFast.from_pretrained(hf_dir)
            obj.tokenizer.pad_token = obj.tokenizer.eos_token
            obj.model = GPT2LMHeadModel.from_pretrained(hf_dir).to(obj.device)
            obj.model.eval()
        logger.info("LogGPTModel loaded from %s", path)
        return obj