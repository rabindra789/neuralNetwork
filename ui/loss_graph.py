"""
LossGraph — live loss curve rendered via matplotlib embedded in PyQt6.

The figure uses a white scientific style matching the brain canvas aesthetic.
It is updated after every training step (pretrain + user learn steps).

Implementation notes:
  • We import matplotlib AFTER QApplication is created (done in main.py).
  • FigureCanvasQTAgg is the official Qt6-compatible canvas class.
  • draw_idle() is used instead of draw() to avoid blocking the event loop.
  • A rolling window of LOSS_GRAPH_MAX_POINTS is kept to prevent unbounded
    memory growth during long demo sessions.
"""

from typing import List

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui     import QFont
from PyQt6.QtCore    import Qt

from config import LOSS_GRAPH_MAX_POINTS


class LossGraph(QWidget):
    """
    Embeds a matplotlib figure showing the training loss curve.
    Call add_loss(value) after every gradient step.
    Call set_history(list) to bulk-load the pre-training history.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._history: List[float] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Title label (rendered by Qt — no matplotlib text overhead)
        hdr = QLabel("Live Training Loss")
        hdr.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        hdr.setStyleSheet("color: #444;")
        hdr.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(hdr)

        # Import matplotlib only when the widget is first constructed
        # (QApplication must already exist at this point)
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(4, 1.4), dpi=90)
        self._fig.patch.set_facecolor("white")

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#fafafa")
        self._ax.tick_params(labelsize=7)
        self._ax.set_xlabel("Step", fontsize=7, color="#555")
        self._ax.set_ylabel("Loss", fontsize=7, color="#555")
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#ddd")

        self._line, = self._ax.plot([], [], color="#1a73e8", linewidth=1.4,
                                    antialiased=True)
        self._fig.tight_layout(pad=0.6)

        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas)

    # ── public API ────────────────────────────────────────────────────────────

    def set_history(self, history: List[float]):
        """Bulk-load pre-training loss history (called after pretrain)."""
        self._history = list(history[-LOSS_GRAPH_MAX_POINTS:])
        self._refresh()

    def add_loss(self, value: float):
        """Append one loss value and redraw."""
        self._history.append(value)
        if len(self._history) > LOSS_GRAPH_MAX_POINTS:
            self._history = self._history[-LOSS_GRAPH_MAX_POINTS:]
        self._refresh()

    # ── internal ──────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self._history:
            return

        xs = np.arange(len(self._history))
        ys = np.array(self._history)

        self._line.set_xdata(xs)
        self._line.set_ydata(ys)

        # Auto-scale with a small top margin
        self._ax.set_xlim(0, max(len(xs) - 1, 1))
        ymin = max(0.0, ys.min() * 0.95)
        ymax = ys.max() * 1.05 + 1e-4
        self._ax.set_ylim(ymin, ymax)

        self._canvas.draw_idle()
