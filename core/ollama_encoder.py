"""
OllamaEncoder — wraps the Ollama embed API for text → dense vector encoding.

Drop-in replacement for the old SemanticEncoder (sentence-transformers).
Uses the local Ollama server (default http://localhost:11434) so no
HuggingFace downloads, no GPU management on our side — Ollama handles it.

The embedding dimension is discovered at init time by probing the model,
so the rest of the codebase stays dimension-agnostic.
"""

import logging
from typing import List

import numpy as np
import ollama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL

log = logging.getLogger(__name__)


class OllamaEncoder:
    """Ollama-backed text encoder with the same API as the old SemanticEncoder."""

    def __init__(self):
        self._client = ollama.Client(host=OLLAMA_BASE_URL)
        self._model = OLLAMA_MODEL

        # Probe the model to discover embedding dimension
        probe = self._client.embed(model=self._model, input="warmup")
        self._dim = len(probe["embeddings"][0])
        log.info("OllamaEncoder ready  [%s  dim=%d]", self._model, self._dim)

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string → (dim,) float32 numpy array."""
        resp = self._client.embed(model=self._model, input=text)
        return np.array(resp["embeddings"][0], dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch-encode a list of strings → (N, dim) float32 numpy array."""
        resp = self._client.embed(model=self._model, input=texts)
        return np.array(resp["embeddings"], dtype=np.float32)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model
