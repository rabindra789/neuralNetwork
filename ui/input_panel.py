"""
InputPanel — left sidebar (v2).

v2 changes:
  • TextEncoder and 8 feature bars removed.
  • Added encoder info label (model name + device).
  • Added memory status label (N/10 interactions stored).
  • update_memory_status(n) exposed for MainWindow to call.
  • All signals and button layout unchanged.
"""

from PyQt6.QtCore    import Qt, pyqtSignal
from PyQt6.QtGui     import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout,
    QPushButton, QLineEdit, QLabel,
    QComboBox, QFrame,
)

from config import CLASS_NAMES, DEVICE, MEMORY_CAPACITY


class InputPanel(QWidget):
    """Left panel.  Emits signals; never calls parent methods directly."""

    process_requested = pyqtSignal(str)          # text
    learn_requested   = pyqtSignal(str, int)     # text, label_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Title
        title = QLabel("Input Module")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #222;")
        root.addWidget(title)

        root.addWidget(self._hline())

        # Text input
        root.addWidget(self._section_label("Enter a sentence:"))
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("e.g. 'Hello, how are you?'")
        self._text_input.setFont(QFont("Segoe UI", 11))
        self._text_input.setFixedHeight(36)
        self._text_input.returnPressed.connect(self._on_process)
        root.addWidget(self._text_input)

        # Class label selector
        root.addWidget(self._section_label("Correct label (for learning):"))
        self._label_combo = QComboBox()
        for name in CLASS_NAMES:
            self._label_combo.addItem(name)
        self._label_combo.setFont(QFont("Segoe UI", 10))
        self._label_combo.setFixedHeight(32)
        root.addWidget(self._label_combo)

        # Buttons
        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)

        self._process_btn = QPushButton("▶  Process Input")
        self._process_btn.setObjectName("ProcessBtn")
        self._process_btn.setFixedHeight(38)
        self._process_btn.clicked.connect(self._on_process)
        btn_col.addWidget(self._process_btn)

        self._learn_btn = QPushButton("⚡  Learn From Input")
        self._learn_btn.setObjectName("LearnBtn")
        self._learn_btn.setFixedHeight(38)
        self._learn_btn.clicked.connect(self._on_learn)
        btn_col.addWidget(self._learn_btn)

        root.addLayout(btn_col)

        root.addWidget(self._hline())

        # Encoder info (replaces the 8 feature bars from v1)
        root.addWidget(self._section_label("Semantic encoder:"))
        device_str = str(DEVICE).upper()
        self._encoder_info = QLabel(
            f"Model: all-MiniLM-L6-v2\n"
            f"Device: {device_str}  |  Dims: 384"
        )
        self._encoder_info.setFont(QFont("Segoe UI", 9))
        self._encoder_info.setStyleSheet(
            "color: #444; background: #f0f6ff; "
            "border-left: 3px solid #1a73e8; "
            "padding: 5px; border-radius: 2px;"
        )
        self._encoder_info.setWordWrap(True)
        root.addWidget(self._encoder_info)

        # Memory status
        root.addWidget(self._section_label("Associative memory:"))
        self._memory_lbl = QLabel(f"0 / {MEMORY_CAPACITY} interactions stored")
        self._memory_lbl.setFont(QFont("Segoe UI", 9))
        self._memory_lbl.setStyleSheet(
            "color: #444; background: #f0fff8; "
            "border-left: 3px solid #00c8b4; "
            "padding: 5px; border-radius: 2px;"
        )
        root.addWidget(self._memory_lbl)

        # Recent memory list
        self._recent_lbl = QLabel("")
        self._recent_lbl.setFont(QFont("Segoe UI", 8))
        self._recent_lbl.setStyleSheet("color: #777; padding-left: 2px;")
        self._recent_lbl.setWordWrap(True)
        root.addWidget(self._recent_lbl)

        root.addStretch()

        # Status
        self._status_lbl = QLabel("Loading model…")
        self._status_lbl.setFont(QFont("Segoe UI", 9))
        self._status_lbl.setStyleSheet("color: #777;")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_process(self):
        text = self._text_input.text().strip()
        if not text:
            self.set_status("Please enter a sentence first.")
            return
        self.process_requested.emit(text)

    def _on_learn(self):
        text  = self._text_input.text().strip()
        label = self._label_combo.currentIndex()
        if not text:
            self.set_status("Please enter a sentence first.")
            return
        self.learn_requested.emit(text, label)

    # ── public API ────────────────────────────────────────────────────────────

    def set_enabled(self, enabled: bool):
        self._process_btn.setEnabled(enabled)
        self._learn_btn.setEnabled(enabled)
        self._text_input.setEnabled(enabled)

    def set_status(self, msg: str):
        self._status_lbl.setText(msg)

    def set_predicted_label(self, class_idx: int):
        if 0 <= class_idx < self._label_combo.count():
            self._label_combo.setCurrentIndex(class_idx)

    def update_memory_status(self, n: int, recent_texts: list = None):
        """Called by MainWindow after each store() or recall()."""
        self._memory_lbl.setText(f"{n} / {MEMORY_CAPACITY} interactions stored")
        if recent_texts:
            lines = "\n".join(f"• {t[:32]}…" if len(t) > 32 else f"• {t}"
                              for t in recent_texts[:3])
            self._recent_lbl.setText(lines)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 9))
        lbl.setStyleSheet("color: #444;")
        return lbl

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #ddd;")
        return line
