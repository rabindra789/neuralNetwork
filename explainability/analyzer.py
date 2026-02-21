"""
PathAnalyzer — identifies the most influential neurons for a prediction.

Algorithm: Greedy Activation-Weighted Path Tracing
─────────────────────────────────────────────────
Starting from the predicted output neuron, we trace backward through the
network layer by layer.  At each step we score every source neuron by:

    score(src) = activation(src) × mean_weight_to_selected_dst_neurons

The top-k neurons at each layer are kept as the 'dominant path'.
This is a simplified version of activation maximisation — lightweight enough
to run interactively without gradient computation.

Why not use gradient-based saliency?
  Gradient computation requires running backward() which would modify the
  training state if the network is in train() mode.  For a live demo we want
  the explanation to be pure inference — no side effects on the model.
"""

from typing import List, Tuple

import numpy as np

from config import VISUAL_LAYER_SIZES


class PathAnalyzer:
    """
    Stateless analyser: call analyse() with a forward pass result and
    receive a list of highlighted (layer_idx, neuron_idx) pairs.
    """

    def __init__(self, top_k: int = 3):
        """
        top_k: number of neurons to highlight per hidden layer.
                Output layer always highlights the winning neuron (top-1).
        """
        self.top_k = top_k

    def analyse(
        self,
        activations:      List[np.ndarray],   # per-layer, shape (n_neurons,)
        weight_magnitudes: List[np.ndarray],   # per-layer-pair, shape (dst, src)
        predicted_class:  int,
    ) -> Tuple[List[Tuple[int, int]], str]:
        """
        Trace the dominant activation path through the network.

        Returns
        -------
        path        : list of (layer_idx, neuron_idx) — neurons to highlight
        explanation : human-readable string for the output panel
        """
        n_layers = len(activations)
        path: List[Tuple[int, int]] = []

        # Output layer: always highlight the winning neuron
        path.append((n_layers - 1, predicted_class))
        selected_dst = [predicted_class]

        # Trace backward from output toward input
        for layer_idx in range(n_layers - 2, -1, -1):
            acts   = activations[layer_idx]           # shape (n_src,)
            w_mat  = weight_magnitudes[layer_idx]     # shape (n_dst, n_src)
            n_src  = len(acts)

            # Score each source neuron: activation × average weight to selected
            scores = np.zeros(n_src)
            for dst_idx in selected_dst:
                if dst_idx < w_mat.shape[0]:
                    scores += w_mat[dst_idx, :n_src] * np.abs(acts)

            # Pick top-k (or fewer if layer is small)
            k            = min(self.top_k, n_src)
            top_indices  = np.argsort(scores)[-k:][::-1]

            for idx in top_indices:
                path.append((layer_idx, int(idx)))

            selected_dst = top_indices.tolist()

        # Build explanation sentence
        explanation = self._build_explanation(activations, predicted_class,
                                              weight_magnitudes)
        return path, explanation

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_explanation(
        activations:      List[np.ndarray],
        predicted_class:  int,
        weight_magnitudes: List[np.ndarray],
    ) -> str:
        from config import CLASS_NAMES

        n_layers    = len(activations)
        input_acts  = activations[0]
        top_feat    = int(np.argmax(input_acts))
        top_feat_v  = float(input_acts[top_feat])

        output_conf = float(activations[-1][predicted_class]) * 100

        # Pick two mid-layer highlights for a richer summary
        mid1_idx = max(1, n_layers // 3)
        mid2_idx = max(mid1_idx + 1, 2 * n_layers // 3)
        h1_top   = int(np.argmax(activations[mid1_idx]))
        h2_top   = int(np.argmax(activations[min(mid2_idx, n_layers - 2)]))

        return (
            f"Dominant semantic dim: #{top_feat} (strength {top_feat_v:.2f}).\n"
            f"Key hidden neurons: L{mid1_idx}[{h1_top}] → L{mid2_idx}[{h2_top}].\n"
            f"Confidence in '{CLASS_NAMES[predicted_class]}': {output_conf:.1f} %."
        )
