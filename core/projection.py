"""
ActivationProjector — maps high-dimensional activations to visual neuron counts.

Problem:
  The real network has layers [384, 256, 128, 64, 32, 12].
  Drawing 256 or 384 neuron circles would be illegible.
  The canvas expects VISUAL_LAYER_SIZES = [40, 40, 40, 40, 32, 12].

Solution — Bucket Averaging:
  Split the activation vector into n_visual equal-width buckets.
  Each visual neuron = mean(|activations| in its bucket).

  This is a form of average pooling:
    • Preserves relative magnitude ordering within a layer.
    • Smooth — nearby neurons in the real network share a visual neuron.
    • Fast: O(n) no matrix multiply.

  Weight projection uses the same principle in 2-D:
    Each visual (dst, src) pair = mean(|weights| in the (dst_bucket, src_bucket)).
    Result is normalised per-layer to [0, 1] for connection thickness mapping.
"""

from typing import List

import numpy as np

from config import ACTUAL_LAYER_SIZES, VISUAL_LAYER_SIZES


class ActivationProjector:
    """
    Projects ACTUAL_LAYER_SIZES activations → VISUAL_LAYER_SIZES for the canvas.
    Also projects weight matrices to match the visual grid.

    Stateless — all methods are deterministic given the same input.
    """

    def __init__(self):
        assert len(ACTUAL_LAYER_SIZES) == len(VISUAL_LAYER_SIZES), \
            "ACTUAL and VISUAL layer lists must have the same length."

    # ── activation projection ─────────────────────────────────────────────────

    def project_activations(self, acts: List[np.ndarray]) -> List[np.ndarray]:
        """
        Project all per-layer activation vectors to their visual sizes.

        acts[i] has shape (ACTUAL_LAYER_SIZES[i],)
        Returns list where result[i] has shape (VISUAL_LAYER_SIZES[i],)
        """
        return [
            self._bucket_avg(a, VISUAL_LAYER_SIZES[i])
            for i, a in enumerate(acts)
        ]

    # ── weight projection ─────────────────────────────────────────────────────

    def project_all_weights(
        self, weight_mags: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Project all weight magnitude matrices to visual grid dimensions.

        weight_mags[i] has shape (ACTUAL_LAYER_SIZES[i+1], ACTUAL_LAYER_SIZES[i])
        Returns list where result[i] has shape (VISUAL_LAYER_SIZES[i+1], VISUAL_LAYER_SIZES[i])
        """
        projected = []
        for i, w in enumerate(weight_mags):
            n_vis_dst = VISUAL_LAYER_SIZES[i + 1]
            n_vis_src = VISUAL_LAYER_SIZES[i]
            projected.append(self._bucket_avg_2d(w, n_vis_dst, n_vis_src))
        return projected

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _bucket_avg(v: np.ndarray, n_buckets: int) -> np.ndarray:
        """
        1-D bucket average of absolute values using np.add.reduceat.

        Expansion (n < n_buckets): nearest-neighbour upsampling.
        Reduction (n > n_buckets): vectorised sum-and-divide per bucket.
        O(n) time, O(n_buckets) space.
        """
        v_abs = np.abs(v).astype(np.float32)
        n = len(v_abs)

        if n == n_buckets:
            return v_abs

        idx = (np.arange(n_buckets, dtype=np.float64) * n / n_buckets).astype(np.intp)

        if n < n_buckets:
            # Expansion: nearest-neighbour (no averaging needed)
            return v_abs[idx]

        # Reduction: vectorised bucket sums then divide by bucket sizes
        sizes = np.diff(np.append(idx, n)).astype(np.float32)
        return (np.add.reduceat(v_abs, idx) / sizes).astype(np.float32)

    @staticmethod
    def _bucket_avg_2d(
        w: np.ndarray, n_dst: int, n_src: int
    ) -> np.ndarray:
        """
        2-D bucket average using two consecutive np.add.reduceat calls.

        Replaces nested Python loops (O(n_dst × n_src) iterations) with
        two C-level numpy passes — ~30-100× faster for large weight matrices.

        Expansion dimensions are handled by nearest-neighbour before reduceat.
        Returns shape (n_dst, n_src), values in [0, 1].
        """
        w_abs = np.abs(w).astype(np.float32)
        orig_dst, orig_src = w_abs.shape

        if orig_dst == n_dst and orig_src == n_src:
            mx = w_abs.max() + 1e-8
            return (w_abs / mx).astype(np.float32)

        # ── Row axis ──────────────────────────────────────────────────────────
        row_idx = (np.arange(n_dst, dtype=np.float64) * orig_dst / n_dst).astype(np.intp)
        if orig_dst < n_dst:
            # Expansion: nearest-neighbour row replication
            w_abs    = w_abs[row_idx, :]
            orig_dst = n_dst
            row_idx  = np.arange(n_dst, dtype=np.intp)

        row_sizes = np.diff(np.append(row_idx, orig_dst)).reshape(-1, 1).astype(np.float32)
        row_avg   = np.add.reduceat(w_abs, row_idx, axis=0) / row_sizes  # (n_dst, orig_src)

        # ── Column axis ───────────────────────────────────────────────────────
        col_idx = (np.arange(n_src, dtype=np.float64) * orig_src / n_src).astype(np.intp)
        if orig_src < n_src:
            # Expansion: nearest-neighbour column replication
            row_avg  = row_avg[:, col_idx]
            orig_src = n_src
            col_idx  = np.arange(n_src, dtype=np.intp)

        col_sizes = np.diff(np.append(col_idx, orig_src)).astype(np.float32)
        result    = (np.add.reduceat(row_avg, col_idx, axis=1) / col_sizes).astype(np.float32)

        # Normalise per-layer so connection thickness maps to [0, 1]
        mx = result.max() + 1e-8
        return (result / mx).astype(np.float32)
