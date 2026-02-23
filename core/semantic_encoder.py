"""
SemanticEncoder — wraps SentenceTransformer "all-MiniLM-L6-v2".

Why this model:
  • 384-dimensional embeddings — rich semantic space, light enough for CPU
  • ~6 M parameters, ~90 MB cached on first run
  • Inference ~80 ms CPU / ~15 ms CUDA for a single sentence
  • Pretrained on 1B+ sentence pairs — generalises to any text without fine-tuning

The encoder is CUDA-aware: if a GPU is available (config.DEVICE), the model is
moved there.  encode() always returns a CPU numpy array regardless of device,
so downstream code is device-agnostic.
"""

import logging
import os
import warnings
from typing import List

# ── Silence ALL HuggingFace / transformers startup noise ──────────────────────
# Must be set before any hf/transformers import to take effect.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # weight-loading bar
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")       # parallelism nag
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")       # LOAD REPORT etc.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")             # HF hub messages

# Suppress the "unauthenticated requests" Python warning
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Set all relevant loggers to ERROR (belt-and-suspenders)
for _log in ("sentence_transformers", "transformers",
             "transformers.modeling_utils", "huggingface_hub",
             "huggingface_hub.utils._headers", "huggingface_hub.file_download"):
    logging.getLogger(_log).setLevel(logging.ERROR)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import DEVICE


class SemanticEncoder:
    """CUDA-aware wrapper around all-MiniLM-L6-v2."""

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        # SentenceTransformer downloads and caches the model on first call.
        # Subsequent calls load from ~/.cache/torch/sentence_transformers/.
        self._model = SentenceTransformer(self._MODEL_NAME, device=str(DEVICE))

        # CUDA warmup — eliminates the 3+ second cold-start on the first real encode.
        # This dummy call initialises CuDNN kernels during startup, not at inference time.
        if DEVICE.type == "cuda":
            _ = self.encode("warmup")

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string → (384,) float32 numpy array.
        Always returns a CPU array so callers need not know the device.
        """
        with torch.inference_mode():
            emb = self._model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        return emb.astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch-encode a list of strings → (N, 384) float32 numpy array.
        Significantly faster than calling encode() in a loop for N > 5.
        """
        with torch.inference_mode():
            embs = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
                batch_size=32,
            )
        return embs.astype(np.float32)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return DEVICE

    @property
    def embedding_dim(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME
