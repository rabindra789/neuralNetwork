"""
ConfidenceGraph — live prediction confidence history chart.

Tracks max(probs) for every inference, coloured by the winning class.
Judges see learning stabilisation and cognitive consistency over time.
"""

from typing import List, Tuple

import numpy as np
from PyQt6.QtCore    import Qt
from PyQt6.QtGui     import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from config import CLASS_NAMES

# Same 12 colours as OutputPanel — shared visual language
_CLASS_HEX = [
    "#4CAF50", "#009688", "#2196F3", "#03A9F4",
    "#9C27B0", "#673AB7", "#FF9800", "#FF5722",
    "#F44336", "#E91E63", "#607D8B", "#795548",
]


class ConfidenceGraph(QWidget):
    """
    Matplotlib line plot showing max confidence per inference over time.
    Each point is coloured by its predicted class.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._history: List[Tuple[int, float]] = []   # (class_idx, max_prob)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        hdr = QLabel("Prediction Confidence History")
        hdr.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        hdr.setStyleSheet("color: #444;")
        hdr.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(hdr)

        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(4, 1.4), dpi=90)
        self._fig.patch.set_facecolor("white")

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#fafafa")
        self._ax.tick_params(labelsize=7)
        self._ax.set_xlabel("Inference", fontsize=7, color="#555")
        self._ax.set_ylabel("Confidence", fontsize=7, color="#555")
        self._ax.set_ylim(0.0, 1.0)
        self._ax.axhline(y=0.40, color="#b0bec5", linewidth=0.8,
                         linestyle="--")   # uncertainty threshold line
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#ddd")

        self._fig.tight_layout(pad=0.6)
        self._canvas_widget = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas_widget)

    # ── public API ────────────────────────────────────────────────────────────

    def add_inference(self, class_idx: int, max_prob: float):
        """Record one inference result and redraw."""
        self._history.append((class_idx, float(max_prob)))
        if len(self._history) > 100:
            self._history = self._history[-100:]
        self._refresh()

    # ── internal ──────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self._history:
            return

        self._ax.cla()
        self._ax.set_facecolor("#fafafa")
        self._ax.tick_params(labelsize=7)
        self._ax.set_xlabel("Inference", fontsize=7, color="#555")
        self._ax.set_ylabel("Confidence", fontsize=7, color="#555")
        self._ax.set_ylim(0.0, 1.05)
        self._ax.axhline(y=0.40, color="#b0bec5", linewidth=0.8,
                         linestyle="--", label="Uncertainty threshold")
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#ddd")

        xs      = np.arange(len(self._history))
        ys      = np.array([p for _, p in self._history])
        colors  = [_CLASS_HEX[c] for c, _ in self._history]

        # Grey line connecting all points
        self._ax.plot(xs, ys, color="#cccccc", linewidth=1.0, zorder=1)

        # Scatter: each point coloured by its class
        self._ax.scatter(xs, ys, c=colors, s=18, zorder=2, linewidths=0)

        # Annotate the last point with class name
        if self._history:
            last_cls, last_prob = self._history[-1]
            label = CLASS_NAMES[last_cls] if last_prob >= 0.40 else "Uncertain"
            self._ax.annotate(
                label,
                xy=(xs[-1], ys[-1]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6,
                color=_CLASS_HEX[last_cls],
            )

        self._ax.set_xlim(-0.5, max(len(xs) - 0.5, 1))
        self._fig.tight_layout(pad=0.6)
        self._canvas_widget.draw_idle()
