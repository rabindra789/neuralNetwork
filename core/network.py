"""
DeepBrainNetwork — PyTorch model: 384 → 256 → 128 → 64 → 32 → 12.

Architecture decisions:
  • LayerNorm after each Linear (preferred over BatchNorm1d for this use-case):
      - Works with batch_size=1, which happens during single-sample online learning.
      - BatchNorm1d requires batch_size > 1 in train() mode.
      - LayerNorm normalises over the feature dimension — same effect, no batch constraint.
  • Dropout (decreasing rate per depth):
      - Reduces overfitting on the small 96-sample training set.
      - Rates: 0.30, 0.25, 0.20, 0.10 — less regularisation near the output.
  • Xavier uniform init: variance-stable gradients from the first epoch.
  • forward() captures per-layer activations post-ReLU (always ≥ 0) and returns
    them alongside the output.  The caller (Trainer) passes these straight to
    the visualisation pipeline — no second forward pass required.
  • All tensors live on config.DEVICE (GPU if available, else CPU).
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ACTUAL_LAYER_SIZES, DEVICE


class DeepBrainNetwork(nn.Module):
    """
    Deep feedforward network:
      Input(384) → FC+LN+ReLU+Drop → 256
                 → FC+LN+ReLU+Drop → 128
                 → FC+LN+ReLU+Drop →  64
                 → FC+LN+ReLU+Drop →  32
                 → FC+Softmax      →  12

    forward() returns (output, activations_list_of_6).
    """

    def __init__(self):
        super().__init__()
        sizes = ACTUAL_LAYER_SIZES    # [384, 256, 128, 64, 32, 12]

        # Fully-connected layers
        self.fc1 = nn.Linear(sizes[0], sizes[1])   # 384 → 256
        self.fc2 = nn.Linear(sizes[1], sizes[2])   # 256 → 128
        self.fc3 = nn.Linear(sizes[2], sizes[3])   # 128 →  64
        self.fc4 = nn.Linear(sizes[3], sizes[4])   #  64 →  32
        self.fc5 = nn.Linear(sizes[4], sizes[5])   #  32 →  12

        # LayerNorm after each hidden Linear — batch-size agnostic (works at bs=1)
        self.bn1 = nn.LayerNorm(sizes[1])
        self.bn2 = nn.LayerNorm(sizes[2])
        self.bn3 = nn.LayerNorm(sizes[3])
        self.bn4 = nn.LayerNorm(sizes[4])

        # Dropout — decreasing rate as we approach the output
        self.drop1 = nn.Dropout(0.30)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.20)
        self.drop4 = nn.Dropout(0.10)

        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self):
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4, self.fc5):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with activation capture.

        Parameters
        ----------
        x : Tensor of shape (batch, 384), on DEVICE

        Returns
        -------
        output      : Softmax probabilities, shape (batch, 12)
        activations : list of 6 detached CPU tensors, shapes:
                      [(batch,384), (batch,256), (batch,128),
                       (batch,64),  (batch,32),  (batch,12)]
        """
        acts: List[torch.Tensor] = []
        acts.append(x.detach().cpu())               # layer 0: raw input embedding

        h1 = self.drop1(F.relu(self.bn1(self.fc1(x))))
        acts.append(h1.detach().cpu())              # layer 1: post-ReLU hidden1

        h2 = self.drop2(F.relu(self.bn2(self.fc2(h1))))
        acts.append(h2.detach().cpu())              # layer 2

        h3 = self.drop3(F.relu(self.bn3(self.fc3(h2))))
        acts.append(h3.detach().cpu())              # layer 3

        h4 = self.drop4(F.relu(self.bn4(self.fc4(h3))))
        acts.append(h4.detach().cpu())              # layer 4

        out = F.softmax(self.fc5(h4), dim=-1)
        acts.append(out.detach().cpu())             # layer 5: output probabilities

        return out, acts

    # ── weight inspection ─────────────────────────────────────────────────────

    def get_weight_magnitudes(self) -> List[np.ndarray]:
        """
        Return per-layer absolute weights normalised to [0, 1].
        Shapes: [(256,384),(128,256),(64,128),(32,64),(12,32)]
        """
        mags = []
        for layer in (self.fc1, self.fc2, self.fc3, self.fc4, self.fc5):
            w   = layer.weight.detach().cpu().numpy()
            w   = np.abs(w)
            mags.append(w / (w.max() + 1e-8))
        return mags

    def get_raw_weights(self) -> List[np.ndarray]:
        """Return unscaled weight matrices (used by Trainer for delta computation)."""
        return [
            layer.weight.detach().cpu().numpy().copy()
            for layer in (self.fc1, self.fc2, self.fc3, self.fc4, self.fc5)
        ]
