"""
BrainMemory — short-term associative memory for the Artificial Brain.

Biological analogy:
  Humans recall similar past experiences when processing new input.
  This module simulates that: when the current sentence is semantically
  similar to a past one, we nudge the input embedding slightly toward the
  memory — the network then processes a mildly "primed" representation.

Algorithm:
  1. store() appends every interaction to a ring buffer (capacity 10).
  2. recall(embedding) computes cosine similarity against all stored embeddings.
  3. If the top match exceeds MEMORY_THRESHOLD, we build a weighted-average
     bias vector from the top-k similar memories.
  4. The bias is scaled by MEMORY_BIAS_SCALE before being added to the input.
     This keeps the perturbation small — it nudges rather than overrides.

Why cosine similarity (not Euclidean distance)?
  Sentence embeddings from all-MiniLM-L6-v2 live on a unit sphere in semantic
  space.  Cosine similarity correctly measures angular distance between meanings,
  while Euclidean distance would be dominated by embedding magnitude.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MEMORY_CAPACITY, MEMORY_THRESHOLD, MEMORY_BIAS_SCALE


class BrainMemory:
    """
    Ring-buffer short-term memory with cosine-similarity recall.

    Each stored entry:
        {
          'embedding': np.ndarray (384,),
          'text':      str,
          'pred':      int  (class index),
          'probs':     np.ndarray (12,),
        }
    """

    def __init__(self, capacity: int = MEMORY_CAPACITY):
        self._ring: deque = deque(maxlen=capacity)
        self._capacity = capacity

    # ── public API ────────────────────────────────────────────────────────────

    def store(
        self,
        embedding: np.ndarray,
        text: str,
        pred: int,
        probs: np.ndarray,
    ) -> None:
        """
        Append an interaction to the ring buffer.
        If full, the oldest entry is automatically dropped (deque maxlen).
        """
        self._ring.append({
            "embedding": embedding.astype(np.float32),
            "text":      text,
            "pred":      int(pred),
            "probs":     probs.astype(np.float32),
        })

    def recall(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
        threshold: float = MEMORY_THRESHOLD,
    ) -> Tuple[Optional[np.ndarray], List[Dict], float]:
        """
        Search memory for past interactions similar to the current embedding.

        Returns
        -------
        bias_vector : np.ndarray (384,) or None
            Weighted-average of top-k similar embeddings × MEMORY_BIAS_SCALE.
            None if no stored memory exceeds the similarity threshold.
        similar     : list of matching memory dicts (sorted by similarity)
        max_sim     : float — highest cosine similarity found (0 if ring empty)
        """
        if not self._ring:
            return None, [], 0.0

        # Compute cosine similarity to every stored embedding
        scored = []
        for mem in self._ring:
            sim = self._cosine_sim(embedding, mem["embedding"])
            scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        max_sim = scored[0][0]

        # Filter by threshold
        matches = [(s, m) for s, m in scored[:top_k] if s >= threshold]
        if not matches:
            return None, [], float(max_sim)

        # Weighted average of matching embeddings
        total_weight = sum(s for s, _ in matches)
        bias = np.zeros_like(embedding)
        for sim, mem in matches:
            bias += (sim / total_weight) * mem["embedding"]

        bias *= MEMORY_BIAS_SCALE
        similar_mems = [m for _, m in matches]
        return bias, similar_mems, float(max_sim)

    def recent_texts(self, n: int = 5) -> List[str]:
        """Return the last n stored texts (newest first)."""
        entries = list(self._ring)[-n:]
        return [e["text"] for e in reversed(entries)]

    def size(self) -> int:
        return len(self._ring)

    @property
    def capacity(self) -> int:
        return self._capacity

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity ∈ [-1, 1].
        Clipped to [0, 1] so negative similarity acts as zero evidence.
        """
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))
