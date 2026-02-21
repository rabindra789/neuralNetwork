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
        1-D bucket average of absolute values.

        If len(v) <= n_buckets: pad with zeros (layer already fits the visual grid).
        Otherwise: split into n_buckets equal-width windows and take the mean.
        Result is NOT normalised here — caller normalises per-layer if needed.
        """
        v_abs = np.abs(v).astype(np.float32)
        n = len(v_abs)

        if n == n_buckets:
            return v_abs

        if n < n_buckets:
            result = np.zeros(n_buckets, dtype=np.float32)
            result[:n] = v_abs
            return result

        result = np.empty(n_buckets, dtype=np.float32)
        for i in range(n_buckets):
            s = int(i * n / n_buckets)
            e = int((i + 1) * n / n_buckets)
            e = max(e, s + 1)   # ensure at least one element
            result[i] = v_abs[s:e].mean()
        return result

    @staticmethod
    def _bucket_avg_2d(
        w: np.ndarray, n_dst: int, n_src: int
    ) -> np.ndarray:
        """
        2-D bucket average of absolute weight values, normalised to [0, 1].

        w has shape (orig_dst, orig_src).
        Returns shape (n_dst, n_src), values in [0, 1].
        """
        w_abs     = np.abs(w).astype(np.float32)
        orig_dst, orig_src = w_abs.shape
        result    = np.empty((n_dst, n_src), dtype=np.float32)

        for di in range(n_dst):
            ds = int(di * orig_dst / n_dst)
            de = int((di + 1) * orig_dst / n_dst)
            de = max(de, ds + 1)
            row_slice = w_abs[ds:de, :]   # shape: (bucket_h, orig_src)

            for si in range(n_src):
                ss = int(si * orig_src / n_src)
                se = int((si + 1) * orig_src / n_src)
                se = max(se, ss + 1)
                result[di, si] = row_slice[:, ss:se].mean()

        # Normalise per-layer so connection thickness maps to [0, 1]
        mx = result.max() + 1e-8
        return result / mx
