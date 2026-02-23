"""
OutputPanel — right sidebar (v2).

v2 changes:
  • 12 confidence bars (one per intent class) with compact 12px height.
  • Memory recall indicator: shows similarity score when memory fires.
  • All other methods unchanged.
"""

from PyQt6.QtCore    import Qt
from PyQt6.QtGui     import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QProgressBar,
)

from config import CLASS_NAMES, UNCERTAINTY_THRESH


# 12 visually distinct colours — one per intent class
_CLASS_COLOURS = [
    "#4CAF50",   # 0  Greeting       — green
    "#009688",   # 1  Farewell       — teal
    "#2196F3",   # 2  What-Q         — blue
    "#03A9F4",   # 3  How-Q          — light blue
    "#9C27B0",   # 4  Help Req.      — purple
    "#673AB7",   # 5  Info Req.      — deep purple
    "#FF9800",   # 6  Confirm        — orange
    "#FF5722",   # 7  Praise         — deep orange
    "#F44336",   # 8  Deny           — red
    "#E91E63",   # 9  Complaint      — pink
    "#607D8B",   # 10 Command        — blue-grey
    "#795548",   # 11 Small Talk     — brown
]


class ConfidenceBar(QWidget):
    """One compact class-label + progress-bar row."""

    def __init__(self, class_name: str, colour: str, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(5)

        lbl = QLabel(class_name)
        lbl.setFixedWidth(78)
        lbl.setFont(QFont("Segoe UI", 8))
        row.addWidget(lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(12)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background: #eeeeee;
                border-radius: 4px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {colour};
                border-radius: 4px;
            }}
        """)

        self._pct = QLabel("0%")
        self._pct.setFixedWidth(28)
        self._pct.setFont(QFont("Segoe UI", 8))
        self._pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row.addWidget(self._bar)
        row.addWidget(self._pct)

    def set_confidence(self, prob: float):
        pct = int(prob * 100)
        self._bar.setValue(pct)
        self._pct.setText(f"{pct}%")


class OutputPanel(QWidget):
    """Right panel — prediction results and memory recall status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)

        # Title
        title = QLabel("Prediction")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #222;")
        root.addWidget(title)

        root.addWidget(self._hline())

        # Predicted class (large display)
        root.addWidget(self._section_label("Predicted intent:"))
        self._class_lbl = QLabel("—")
        self._class_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self._class_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._class_lbl.setStyleSheet(
            "color: #1a73e8; background: #f0f6ff; "
            "border-radius: 8px; padding: 8px;"
        )
        self._class_lbl.setFixedHeight(52)
        root.addWidget(self._class_lbl)

        # Memory recall indicator
        self._memory_recall_lbl = QLabel("")
        self._memory_recall_lbl.setFont(QFont("Segoe UI", 8))
        self._memory_recall_lbl.setStyleSheet(
            "color: #00695c; background: #e0f2f1; "
            "border-left: 3px solid #00c8b4; "
            "padding: 3px 5px; border-radius: 2px;"
        )
        self._memory_recall_lbl.setVisible(False)
        root.addWidget(self._memory_recall_lbl)

        # 12 confidence bars
        root.addWidget(self._section_label("Class confidence:"))
        self._bars: list[ConfidenceBar] = []
        for name, colour in zip(CLASS_NAMES, _CLASS_COLOURS):
            bar = ConfidenceBar(name, colour)
            root.addWidget(bar)
            self._bars.append(bar)

        root.addWidget(self._hline())

        # Reasoning
        root.addWidget(self._section_label("Reasoning:"))
        self._explain_lbl = QLabel("Run inference to see an explanation.")
        self._explain_lbl.setFont(QFont("Segoe UI", 9))
        self._explain_lbl.setStyleSheet(
            "color: #444; background: #fafafa; "
            "border-left: 3px solid #1a73e8; "
            "padding: 5px; border-radius: 2px;"
        )
        self._explain_lbl.setWordWrap(True)
        self._explain_lbl.setMinimumHeight(60)
        root.addWidget(self._explain_lbl)

        root.addStretch()

        self._step_lbl = QLabel("Training steps: 0")
        self._step_lbl.setFont(QFont("Segoe UI", 8))
        self._step_lbl.setStyleSheet("color: #999;")
        root.addWidget(self._step_lbl)

    # ── public API ────────────────────────────────────────────────────────────

    def set_prediction(self, class_idx: int, probs):
        max_prob = float(max(probs))
        if max_prob < UNCERTAINTY_THRESH:
            # Brain is not confident enough — show uncertain state
            self._class_lbl.setText("Uncertain")
            self._class_lbl.setStyleSheet(
                "color: #78909c; background: #f5f5f5; "
                "border-radius: 8px; padding: 8px; border: 2px dashed #90a4ae;"
            )
        else:
            name   = CLASS_NAMES[class_idx]
            colour = _CLASS_COLOURS[class_idx]
            self._class_lbl.setText(name)
            self._class_lbl.setStyleSheet(
                f"color: {colour}; background: #f8f8f8; "
                f"border-radius: 8px; padding: 8px; border: 2px solid {colour};"
            )
        for i, bar in enumerate(self._bars):
            bar.set_confidence(float(probs[i]))

    def set_memory_recall(self, similarity: float):
        """Show teal banner when memory recall fires."""
        pct = int(similarity * 100)
        self._memory_recall_lbl.setText(
            f"🧠  Memory recall triggered  ({pct}% similarity)"
        )
        self._memory_recall_lbl.setVisible(True)

    def clear_memory_recall(self):
        self._memory_recall_lbl.setVisible(False)

    def set_explanation(self, text: str):
        self._explain_lbl.setText(text)

    def set_step_count(self, steps: int):
        self._step_lbl.setText(f"Training steps: {steps}")

    def reset(self):
        self._class_lbl.setText("—")
        self._class_lbl.setStyleSheet(
            "color: #1a73e8; background: #f0f6ff; "
            "border-radius: 8px; padding: 8px;"
        )
        for bar in self._bars:
            bar.set_confidence(0.0)
        self._explain_lbl.setText("Run inference to see an explanation.")
        self._memory_recall_lbl.setVisible(False)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet("color: #555;")
        return lbl

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #ddd;")
        return line
