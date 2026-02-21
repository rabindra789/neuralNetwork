"""
NeuronItem — Visual representation of a single neuron.

Design contract:
  - activation  [0, 1]  → fill colour lerps from inactive-white to active-blue
  - highlighted  bool   → gold ring drawn outside, used by explainability module
  - paint() is stateless w.r.t. canvas; all animation state is passed as args
"""

from PyQt6.QtCore  import Qt, QRectF
from PyQt6.QtGui   import QPainter, QPen, QBrush, QColor

from config import (
    NEURON_RADIUS,
    COLOR_NEURON_INACTIVE, COLOR_NEURON_ACTIVE,
    COLOR_NEURON_BORDER,   COLOR_BACKPROP, COLOR_HIGHLIGHT, COLOR_MEMORY,
)


class NeuronItem:
    """
    Data + drawing object for one neuron circle.

    Coordinates (x, y) are floats in canvas-pixel space.
    All colour computation is done in paint() so the item stays
    serialisable / copyable with no Qt dependency at construction time.
    """

    def __init__(self, x: float, y: float, layer_idx: int, neuron_idx: int):
        self.x           = x
        self.y           = y
        self.layer_idx   = layer_idx
        self.neuron_idx  = neuron_idx
        self.radius      = NEURON_RADIUS      # may be overridden by canvas for dense layers
        self.activation  = 0.0               # real value from forward pass, normalised [0,1]
        self.highlighted = False             # set by explainability analyser

    # ── public API called by BrainCanvas ────────────────────────────────────

    def set_activation(self, value: float):
        """Clamp activation to [0, 1].  Negative activations (pre-ReLU) map to 0."""
        self.activation = max(0.0, min(1.0, float(value)))

    def reset(self):
        self.activation  = 0.0
        self.highlighted = False

    # ── drawing ─────────────────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        backprop_alpha: float = 0.0,
        memory_alpha:   float = 0.0,
    ):
        """
        Draw the neuron at (self.x, self.y).

        backprop_alpha [0,1] — orange tint: gradient just flowed through here.
        memory_alpha   [0,1] — teal tint: associative memory influenced this neuron.
        """
        cx, cy, r = self.x, self.y, float(self.radius)
        t          = self.activation

        # ── Base colour: lerp inactive (white) → active (blue) ────────────
        ir, ig, ib = COLOR_NEURON_INACTIVE
        ar, ag, ab = COLOR_NEURON_ACTIVE
        fr = int(ir + (ar - ir) * t)
        fg = int(ig + (ag - ig) * t)
        fb = int(ib + (ab - ib) * t)

        # Blend toward orange for backprop flash
        if backprop_alpha > 0.0:
            br_c, bg_c, bb_c = COLOR_BACKPROP
            b  = backprop_alpha
            fr = int(fr * (1.0 - b) + br_c * b)
            fg = int(fg * (1.0 - b) + bg_c * b)
            fb = int(fb * (1.0 - b) + bb_c * b)

        # Blend toward teal for memory recall (overrides backprop when present)
        if memory_alpha > 0.0:
            mr, mg, mb = COLOR_MEMORY
            m  = memory_alpha
            fr = int(fr * (1.0 - m) + mr * m)
            fg = int(fg * (1.0 - m) + mg * m)
            fb = int(fb * (1.0 - m) + mb * m)

        fr = max(0, min(255, fr))
        fg = max(0, min(255, fg))
        fb = max(0, min(255, fb))

        # ── Glow halo (blue) ──────────────────────────────────────────────
        if t > 0.12:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(ar, ag, ab, int(55 * t))))
            gr = r + 6
            painter.drawEllipse(QRectF(cx - gr, cy - gr, gr * 2, gr * 2))

        # ── Memory recall glow (teal halo) ────────────────────────────────
        if memory_alpha > 0.05:
            mr, mg, mb = COLOR_MEMORY
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(mr, mg, mb, int(70 * memory_alpha))))
            gr = r + 8
            painter.drawEllipse(QRectF(cx - gr, cy - gr, gr * 2, gr * 2))

        # ── Explainability highlight ring (gold) ──────────────────────────
        if self.highlighted:
            hr_c, hg_c, hb_c = COLOR_HIGHLIGHT
            painter.setPen(QPen(QColor(hr_c, hg_c, hb_c), 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            hr = r + 5
            painter.drawEllipse(QRectF(cx - hr, cy - hr, hr * 2, hr * 2))

        # ── Main neuron circle ─────────────────────────────────────────────
        br_n, bg_n, bb_n = COLOR_NEURON_BORDER
        painter.setBrush(QBrush(QColor(fr, fg, fb)))
        painter.setPen(QPen(QColor(br_n, bg_n, bb_n), 1.5))
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
