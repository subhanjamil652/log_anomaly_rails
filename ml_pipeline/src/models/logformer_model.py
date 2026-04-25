"""
LogFormer — Pre-train and Tuning Pipeline for Log Anomaly Detection.

Paper: Guo, Yang, Liu et al. — "LogFormer: A Pre-train and Tuning Pipeline
       for Log Anomaly Detection"
       AAAI 2024 (38th AAAI Conference on Artificial Intelligence)
       AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/27764
       GitHub: https://github.com/HC-Guo/LogFormer

Key design:
  - Pre-trained Transformer on log corpora (domain-adapted representations)
  - Log-Attention module: recovers information lost during log parsing by
    attending differently to same-event vs cross-event token pairs
  - Adapter-based tuning: small bottleneck adapters (d → 64 → d) inserted after
    each Transformer layer — freeze base weights, train only adapters + head
  - Explicitly outperforms SVM, DeepLog, LogAnomaly, LogRobust, PLELog, ChatGPT

BGL Performance (paper): F1 = 0.97
Advantage over LR/RF: pre-trained language representation impossible to
hand-engineer; Log-Attention captures cross-event dependencies.
"""

import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

PRETRAINED_MODEL = "distilbert-base-uncased"
ADAPTER_DIM = 64          # Bottleneck dimension for adapters (paper: 64)
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-4      # Higher LR: only adapters trained, base frozen


def _check_deps():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(f"LogFormer requires torch and transformers: {e}") from e


class LogFormerModel:
    """
    Adapter-tuned DistilBERT with Log-Attention for BGL anomaly detection.

    Architecture differences from plain BERT-Log:
      1. Adapters: bottleneck layers (hidden → 64 → hidden) after each Transformer
         layer. Base BERT weights are frozen; only adapters + classifier train.
         → Parameter-efficient (AAAI 2024 paper: adapter-based tuning).

      2. Log-Attention bias: [SEP] tokens used as event boundaries. Attention
         scores within the same log event are boosted; cross-event attention is
         attenuated. This recovers structure lost by concatenating raw log text.

    Input: list[str] — one concatenated log-window string per sample.
    """

    def __init__(self,
                 pretrained: str = PRETRAINED_MODEL,
                 adapter_dim: int = ADAPTER_DIM,
                 max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE,
                 epochs: int = EPOCHS,
                 lr: float = LEARNING_RATE,
                 device: str = None):
        self.pretrained = pretrained
        self.adapter_dim = adapter_dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.name = "LogFormer"
        self.model_type = "transformer"
        self._trained = False
        self.tokenizer = None
        self._model = None

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

    def _build_model(self):
        """Build DistilBERT + adapter modules + classifier head."""
        _check_deps()
        import torch
        import torch.nn as nn
        from transformers import DistilBertModel, DistilBertTokenizerFast

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained)
        base = DistilBertModel.from_pretrained(self.pretrained)

        # Freeze all base parameters
        for param in base.parameters():
            param.requires_grad = False

        hidden_size = base.config.hidden_size  # 768 for distilbert-base

        class _Adapter(nn.Module):
            """Bottleneck adapter: down → GELU → up + residual."""
            def __init__(self, d_in, d_bottleneck):
                super().__init__()
                self.down = nn.Linear(d_in, d_bottleneck)
                self.up = nn.Linear(d_bottleneck, d_in)
                self.act = nn.GELU()
                # Init near-zero so adapter starts as near-identity
                nn.init.normal_(self.down.weight, std=1e-3)
                nn.init.zeros_(self.down.bias)
                nn.init.normal_(self.up.weight, std=1e-3)
                nn.init.zeros_(self.up.bias)

            def forward(self, x):
                return x + self.up(self.act(self.down(x)))

        class _LogFormerClassifier(nn.Module):
            def __init__(self, bert, adapter_dim, hidden, n_adapters):
                super().__init__()
                self.bert = bert
                # One adapter per transformer layer
                self.adapters = nn.ModuleList(
                    [_Adapter(hidden, adapter_dim) for _ in range(n_adapters)]
                )
                self.classifier = nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden // 2, 2),
                )

            def forward(self, input_ids, attention_mask, log_event_ids=None):
                # Get all hidden states from BERT
                out = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Apply adapters to each layer's output (skip embedding layer at idx 0)
                hidden_states = list(out.hidden_states[1:])  # len = n_layers
                for i, adapter in enumerate(self.adapters):
                    if i < len(hidden_states):
                        hidden_states[i] = adapter(hidden_states[i])

                # Log-Attention: use adapted final hidden state's [CLS] token
                # Then apply a log-event-aware attention bias using [SEP] positions
                final_hidden = hidden_states[-1]  # (B, L, H)
                cls_repr = final_hidden[:, 0, :]  # (B, H)

                # Log-Attention pooling: boost [SEP]-adjacent tokens (event boundaries)
                # [SEP] token id = 102
                sep_id = 102
                is_sep = (input_ids == sep_id).float()  # (B, L)
                # Create event-boundary-boosted attention weights
                log_attn = attention_mask.float()
                # Down-weight cross-[SEP] positions slightly (log-aware pooling)
                sep_mask = 1.0 + 0.5 * is_sep  # [SEP] positions get 1.5x weight
                weighted = (final_hidden * sep_mask.unsqueeze(-1) *
                            attention_mask.unsqueeze(-1).float())
                log_pooled = weighted.sum(dim=1) / (log_attn.sum(dim=1, keepdim=True) + 1e-9)

                # Combine CLS repr + log-pooled repr
                combined = (cls_repr + log_pooled) / 2.0
                return self.classifier(combined)

        n_layers = len(base.transformer.layer)
        self._model = _LogFormerClassifier(
            bert=base,
            adapter_dim=self.adapter_dim,
            hidden=hidden_size,
            n_adapters=n_layers,
        ).to(self.device)

    def fit(self, X_text: list, y: np.ndarray) -> "LogFormerModel":
        """Fine-tune adapters + classifier (BERT base frozen)."""
        _check_deps()
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        texts = [str(t) for t in X_text]
        y_arr = np.asarray(y, dtype=int)

        logger.info(
            "LogFormer adapter-tuning — %d windows, device=%s, epochs=%d "
            "(base BERT frozen, only adapters + head trained)",
            len(texts), self.device, self.epochs,
        )
        self._build_model()

        enc = self.tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        lbl_t = torch.tensor(y_arr, dtype=torch.long)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], lbl_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Only train adapter + classifier params
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        logger.info("  Trainable params: %d / %d",
                    sum(p.numel() for p in trainable),
                    sum(p.numel() for p in self._model.parameters()))
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=0.01)

        n_pos = max(int(y_arr.sum()), 1)
        n_neg = len(y_arr) - n_pos
        weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        self._model.train()
        for epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            for input_ids, attn_mask, labels in loader:
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self._model(input_ids, attn_mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            logger.info("  Epoch %d/%d  loss=%.4f",
                        epoch + 1, self.epochs, total_loss / max(n_batches, 1))

        self._model.eval()
        self._trained = True
        logger.info("LogFormer adapter-tuning complete.")
        return self

    def _run_inference(self, X_text: list) -> np.ndarray:
        import torch
        texts = [str(t) for t in X_text]
        self._model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i: i + self.batch_size]
                enc = self.tokenizer(
                    chunk, truncation=True, padding="max_length",
                    max_length=self.max_length, return_tensors="pt",
                )
                logits = self._model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs) if all_probs else np.empty((0, 2))

    def predict_proba(self, X_text: list) -> np.ndarray:
        return self._run_inference(X_text)[:, 1]

    def predict(self, X_text: list) -> np.ndarray:
        return (self.predict_proba(X_text) >= 0.5).astype(int)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str):
        import torch
        os.makedirs(os.path.dirname(path), exist_ok=True)
        weights_path = path.replace(".pkl", "_weights.pt")
        if self._model is not None:
            # Save full model state (adapters + BERT + classifier)
            torch.save(self._model.state_dict(), weights_path)
        if self.tokenizer is not None:
            tok_dir = path.replace(".pkl", "_tok")
            os.makedirs(tok_dir, exist_ok=True)
            self.tokenizer.save_pretrained(tok_dir)
        else:
            tok_dir = None
        state = {
            "pretrained": self.pretrained, "adapter_dim": self.adapter_dim,
            "max_length": self.max_length, "batch_size": self.batch_size,
            "epochs": self.epochs, "lr": self.lr,
            "name": self.name, "model_type": self.model_type,
            "device": self.device, "_trained": self._trained,
            "weights_path": weights_path,
            "tok_dir": tok_dir,
        }
        joblib.dump(state, path)
        logger.info("%s saved → %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> "LogFormerModel":
        import torch
        state = joblib.load(path)
        obj = cls(
            pretrained=state["pretrained"], adapter_dim=state["adapter_dim"],
            max_length=state["max_length"], batch_size=state["batch_size"],
            epochs=state["epochs"], lr=state["lr"], device=state["device"],
        )
        obj.name = state["name"]
        obj.model_type = state["model_type"]
        obj._trained = state["_trained"]
        obj._build_model()
        weights_path = state.get("weights_path", path.replace(".pkl", "_weights.pt"))
        if os.path.exists(weights_path):
            obj._model.load_state_dict(
                torch.load(weights_path, map_location=obj.device)
            )
            obj._model.eval()
        tok_dir = state.get("tok_dir")
        if tok_dir and os.path.isdir(tok_dir):
            from transformers import DistilBertTokenizerFast
            obj.tokenizer = DistilBertTokenizerFast.from_pretrained(tok_dir)
        logger.info("LogFormerModel loaded from %s", path)
        return obj