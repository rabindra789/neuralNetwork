"""
ConnectionItem — Visual representation of a synaptic connection (weight edge).

Design contract:
  - weight_magnitude [0, 1]  → line opacity and thickness
  - paint() receives pulse_t and backprop_alpha from the canvas animation state
  - pulse_t [0, 1] → position of the travelling blue signal dot along the line
  - backprop_alpha [0, 1] → orange tint intensity during learning flash
"""

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui  import QPainter, QPen, QBrush, QColor

from config import (
    COLOR_CONN_WEAK, COLOR_CONN_STRONG,
    COLOR_PULSE, COLOR_BACKPROP,
)

# Pulse dot size in pixels
_PULSE_RADIUS = 4.5

# Connection line thickness range (px)
_MIN_THICKNESS = 0.4
_MAX_THICKNESS = 2.8


class ConnectionItem:
    """
    Visual edge between two NeuronItem objects.

    weight_magnitude is set externally by BrainCanvas.update_weights()
    whenever the PyTorch model weights change.
    """

    def __init__(self, src_neuron, dst_neuron, layer_pair_idx: int):
        self.src         = src_neuron
        self.dst         = dst_neuron
        self.layer_pair  = layer_pair_idx
        self.weight_magnitude = 0.3   # normalised [0, 1]; updated from model weights

    # ── drawing ─────────────────────────────────────────────────────────────

    def paint(self,
              painter: QPainter,
              pulse_t: float | None = None,
              backprop_alpha: float = 0.0):
        """
        Draw the synaptic connection line, optionally with a travelling pulse
        dot (forward pass) or an orange tint overlay (backprop).

        pulse_t        None  → no pulse drawn
                       0..1  → blue dot rendered at this fraction along the line
        backprop_alpha 0..1  → orange overlay intensity
        """
        sx, sy = self.src.x, self.src.y
        dx, dy = self.dst.x, self.dst.y
        w      = self.weight_magnitude   # 0..1

        # ── Line colour: lerp between weak (light gray) and strong (dark gray) ─
        wr, wg, wb = COLOR_CONN_WEAK
        sr, sg, sb = COLOR_CONN_STRONG
        cr = int(wr + (sr - wr) * w)
        cg = int(wg + (sg - wg) * w)
        cb = int(wb + (sb - wb) * w)

        # Base opacity: strong connections are fully opaque, weak ones are faint
        base_alpha = int(40 + w * 180)   # range 40..220

        # Backprop: blend toward orange and increase opacity
        if backprop_alpha > 0.0:
            br, bg, bb = COLOR_BACKPROP
            b  = backprop_alpha
            cr = int(cr * (1.0 - b) + br * b)
            cg = int(cg * (1.0 - b) + bg * b)
            cb = int(cb * (1.0 - b) + bb * b)
            base_alpha = int(base_alpha * (1.0 - b) + 230 * b)

        thickness = _MIN_THICKNESS + w * (_MAX_THICKNESS - _MIN_THICKNESS)
        if backprop_alpha > 0.0:
            thickness = max(thickness, backprop_alpha * _MAX_THICKNESS)

        # ── Draw the line ──────────────────────────────────────────────────
        painter.setPen(QPen(QColor(cr, cg, cb, base_alpha), thickness))
        painter.drawLine(QPointF(sx, sy), QPointF(dx, dy))

        # ── Draw travelling pulse dot ──────────────────────────────────────
        if pulse_t is not None and 0.0 <= pulse_t <= 1.0:
            self.paint_dot(painter, pulse_t)

    def paint_dot(self, painter: QPainter, pulse_t: float):
        """Draw only the travelling pulse dot. Used by the fast pixmap paint path."""
        if not (0.0 <= pulse_t <= 1.0):
            return
        sx, sy = self.src.x, self.src.y
        dx, dy = self.dst.x, self.dst.y
        px = sx + (dx - sx) * pulse_t
        py = sy + (dy - sy) * pulse_t
        pr, pg, pb = COLOR_PULSE
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(pr, pg, pb, 220)))
        painter.drawEllipse(
            QRectF(px - _PULSE_RADIUS, py - _PULSE_RADIUS,
                   _PULSE_RADIUS * 2,  _PULSE_RADIUS * 2)
        )
