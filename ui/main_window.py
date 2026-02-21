"""
MainWindow — application shell and signal/slot orchestration hub (v2).

v2 changes:
  • PretrainWorker → StartupWorker:
      Step 1: loads SemanticEncoder (may download ~90 MB on first run)
      Step 2: creates Trainer(encoder) and calls pretrain()
      finished signal carries (encoder, trainer) objects back to main thread.

  • MainWindow owns:
      self._encoder    : SemanticEncoder
      self._trainer    : Trainer
      self._memory     : BrainMemory
      self._projector  : ActivationProjector

  • PROCESS/LEARN paths now inject the projector:
      • Encode text → get raw embedding
      • Ask BrainMemory for recall bias + max_sim
      • Run trainer.infer(text, memory_bias)
      • Store interaction in memory
      • Project activations and weight magnitudes → visual sizes
      • Pass visual arrays to canvas (canvas stays blissfully unaware of real dims)
      • If max_sim > threshold: flash canvas input layer in teal

  • Canvas signal memory_recall_occurred connected to output panel banner.
"""

from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui     import QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QSizePolicy,
    QProgressDialog, QApplication,
)

from config import PRETRAIN_EPOCHS, MEMORY_THRESHOLD
from core.memory               import BrainMemory
from core.projection           import ActivationProjector
from visualization.brain_canvas import BrainCanvas
from ui.input_panel             import InputPanel
from ui.output_panel            import OutputPanel
from ui.loss_graph              import LossGraph
from explainability.analyzer    import PathAnalyzer


# ── Startup worker ────────────────────────────────────────────────────────────

class StartupWorker(QObject):
    """
    Runs off the main thread:
      1. Instantiates SemanticEncoder  (may trigger ~90 MB model download)
      2. Creates Trainer(encoder)
      3. Calls trainer.pretrain()
    Emits finished(encoder, trainer) when done.
    """
    progress = pyqtSignal(str, int)          # message, percentage 0..100
    finished = pyqtSignal(object, object)    # encoder, trainer
    error    = pyqtSignal(str)

    def run(self):
        try:
            # Step 1: load SentenceTransformer (cached after first run)
            self.progress.emit("Loading SentenceTransformer model…", 5)
            from core.semantic_encoder import SemanticEncoder
            encoder = SemanticEncoder()
            self.progress.emit(
                f"Encoder ready  [{encoder.model_name}  on {encoder.device}]", 20
            )

            # Step 2: build trainer
            from core.trainer import Trainer
            trainer = Trainer(encoder)
            self.progress.emit("Pre-training deep network (300 epochs)…", 22)

            def cb(epoch: int, loss: float):
                pct = 22 + int(epoch / PRETRAIN_EPOCHS * 76)
                self.progress.emit(
                    f"Training epoch {epoch}/{PRETRAIN_EPOCHS}   loss={loss:.4f}", pct
                )

            trainer.pretrain(progress_cb=cb)
            self.progress.emit("Ready!", 100)
            self.finished.emit(encoder, trainer)

        except Exception as exc:
            self.error.emit(str(exc))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Visual Neural Network"
        )
        self.setMinimumSize(1440, 760)

        # Populated in _on_startup_done()
        self._encoder   = None
        self._trainer   = None
        self._memory    = BrainMemory()
        self._projector = ActivationProjector()
        self._analyzer  = PathAnalyzer(top_k=3)

        self._busy          = False
        self._pending_learn = None   # (text, label) deferred to after forward anim

        # Cached per-inference results (used across signal callbacks)
        self._last_visual_acts = None
        self._last_pred        = None
        self._last_probs       = None

        self._build_ui()
        self._connect_signals()
        self._apply_stylesheet()
        self._start_startup()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(48)
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(18, 0, 18, 0)
        t = QLabel("")
        t.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        t.setObjectName("HeaderTitle")
        hrow.addWidget(t)
        hrow.addStretch()
        sub = QLabel(
            "SentenceTransformer · 384-dim · 6-layer deep net · "
            "Associative Memory · Real-time PyTorch"
        )
        sub.setFont(QFont("Segoe UI", 9))
        sub.setObjectName("HeaderSub")
        hrow.addWidget(sub)
        outer.addWidget(header)

        # Working area
        work_row = QHBoxLayout()
        work_row.setContentsMargins(0, 0, 0, 0)
        work_row.setSpacing(0)

        self._input_panel = InputPanel()
        self._input_panel.setObjectName("SidePanel")
        work_row.addWidget(self._input_panel)
        work_row.addWidget(self._vline())

        self._canvas = BrainCanvas()
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        work_row.addWidget(self._canvas)
        work_row.addWidget(self._vline())

        self._output_panel = OutputPanel()
        self._output_panel.setObjectName("SidePanel")
        work_row.addWidget(self._output_panel)

        outer.addLayout(work_row, stretch=1)

        outer.addWidget(self._hline())
        self._loss_graph = LossGraph()
        self._loss_graph.setObjectName("LossPanel")
        outer.addWidget(self._loss_graph)

    # ── signal connections ────────────────────────────────────────────────────

    def _connect_signals(self):
        self._input_panel.process_requested.connect(self._on_process)
        self._input_panel.learn_requested.connect(self._on_learn)
        self._canvas.forward_animation_done.connect(self._on_forward_done)
        self._canvas.backprop_animation_done.connect(self._on_backprop_done)
        self._canvas.memory_recall_occurred.connect(self._on_memory_recall)

    # ── startup ───────────────────────────────────────────────────────────────

    def _start_startup(self):
        dlg = QProgressDialog(
            "Initialising Artificial Brain…", None, 0, 100, self
        )
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        self._startup_dlg = dlg

        self._startup_thread = QThread()
        self._startup_worker = StartupWorker()
        self._startup_worker.moveToThread(self._startup_thread)
        self._startup_thread.started.connect(self._startup_worker.run)
        self._startup_worker.progress.connect(self._on_startup_progress)
        self._startup_worker.finished.connect(self._on_startup_done)
        self._startup_worker.finished.connect(self._startup_thread.quit)
        self._startup_worker.error.connect(self._on_startup_error)
        self._startup_thread.start()

    def _on_startup_progress(self, msg: str, pct: int):
        self._startup_dlg.setValue(pct)
        self._startup_dlg.setLabelText(msg)
        QApplication.processEvents()

    def _on_startup_done(self, encoder, trainer):
        self._startup_dlg.close()
        self._encoder = encoder
        self._trainer = trainer

        # Initialise canvas with projected weights
        wm         = self._trainer.get_weight_magnitudes()
        visual_wm  = self._projector.project_all_weights(wm)
        self._canvas.update_weights(visual_wm)

        self._loss_graph.set_history(self._trainer.loss_history)
        self._input_panel.set_enabled(True)
        self._input_panel.set_status(
            "Ready. Type a sentence and press Process."
        )

    def _on_startup_error(self, msg: str):
        self._startup_dlg.close()
        self._input_panel.set_status(f"Startup error: {msg}")

    # ── PROCESS path ─────────────────────────────────────────────────────────

    def _on_process(self, text: str):
        if self._busy or self._trainer is None:
            return
        self._set_busy(True)
        self._pending_learn = None
        self._output_panel.clear_memory_recall()

        # 1. Encode raw embedding
        embedding = self._encoder.encode(text)

        # 2. Memory recall (may return None bias)
        bias, similar, max_sim = self._memory.recall(embedding)

        # 3. Forward pass (with optional memory bias)
        probs, acts, pred, _ = self._trainer.infer(text, memory_bias=bias)

        # 4. Store in memory AFTER inference
        self._memory.store(embedding, text, pred, probs)

        # 5. Project activations + weights → visual sizes
        visual_acts = self._projector.project_activations(acts)
        visual_wm   = self._projector.project_all_weights(
            self._trainer.get_weight_magnitudes()
        )

        # 6. Update UI immediately (output + memory status)
        self._output_panel.set_prediction(pred, probs)
        self._input_panel.set_predicted_label(pred)
        self._input_panel.update_memory_status(
            self._memory.size(), self._memory.recent_texts(3)
        )
        self._input_panel.set_status("Propagating signal through network…")

        # 7. Trigger memory flash if recall happened
        if max_sim >= MEMORY_THRESHOLD:
            self._canvas.flash_memory_recall(max_sim)

        # 8. Start animation
        self._canvas.animate_forward_pass(visual_acts, visual_wm)

        # Cache for explainability
        self._last_visual_acts = visual_acts
        self._last_pred        = pred
        self._last_probs       = probs

    def _on_forward_done(self):
        if self._pending_learn is not None:
            self._on_learn_forward_done()
            return
        self._run_explainability()
        self._set_busy(False)
        self._input_panel.set_status("Done. Try another sentence or press Learn.")

    # ── LEARN path ───────────────────────────────────────────────────────────

    def _on_learn(self, text: str, label: int):
        if self._busy or self._trainer is None:
            return
        self._set_busy(True)
        self._pending_learn = (text, label)
        self._output_panel.clear_memory_recall()

        embedding = self._encoder.encode(text)
        bias, similar, max_sim = self._memory.recall(embedding)
        probs, acts, pred, _   = self._trainer.infer(text, memory_bias=bias)

        visual_acts = self._projector.project_activations(acts)
        visual_wm   = self._projector.project_all_weights(
            self._trainer.get_weight_magnitudes()
        )

        self._output_panel.set_prediction(pred, probs)
        self._input_panel.set_status("Forward pass… then learning…")

        if max_sim >= MEMORY_THRESHOLD:
            self._canvas.flash_memory_recall(max_sim)

        self._canvas.animate_forward_pass(visual_acts, visual_wm)

        self._last_visual_acts = visual_acts
        self._last_pred        = pred
        self._last_probs       = probs

    def _on_learn_forward_done(self):
        text, label = self._pending_learn
        self._pending_learn = None

        # Actual gradient step
        loss    = self._trainer.learn_from_input(text, label)
        new_wm  = self._trainer.get_weight_magnitudes()
        vis_wm  = self._projector.project_all_weights(new_wm)

        self._loss_graph.add_loss(loss)
        self._output_panel.set_step_count(len(self._trainer.loss_history))
        self._input_panel.set_status(f"Backpropagating… loss={loss:.4f}")

        # Store the post-learn interaction in memory
        embedding = self._encoder.encode(text)
        self._memory.store(embedding, text, label,
                           self._last_probs if self._last_probs is not None
                           else [0.0] * 12)
        self._input_panel.update_memory_status(
            self._memory.size(), self._memory.recent_texts(3)
        )

        self._canvas.update_weights(vis_wm)
        self._canvas.animate_backprop(self._trainer.weight_deltas)

    def _on_backprop_done(self):
        self._run_explainability()
        self._set_busy(False)
        self._input_panel.set_status("Network updated. Weights adjusted.")

    # ── memory recall banner ──────────────────────────────────────────────────

    def _on_memory_recall(self, similarity: float):
        self._output_panel.set_memory_recall(similarity)

    # ── explainability ────────────────────────────────────────────────────────

    def _run_explainability(self):
        if self._last_visual_acts is None:
            return
        vis_wm = self._projector.project_all_weights(
            self._trainer.get_weight_magnitudes()
        )
        path, explanation = self._analyzer.analyse(
            self._last_visual_acts, vis_wm, self._last_pred
        )
        self._canvas.highlight_neurons(path)
        self._output_panel.set_explanation(explanation)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool):
        self._busy = busy
        self._input_panel.set_enabled(not busy)
        if not busy:
            self._canvas.clear_highlights()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow  { background: #f4f4f4; }
            QFrame#Header {
                background: #1a1a2e;
                border-bottom: 1px solid #0d0d1e;
            }
            QLabel#HeaderTitle { color: #e8eaf6; }
            QLabel#HeaderSub   { color: #7986cb; }
            QWidget#SidePanel  { background: #ffffff; border: none; }
            QWidget#LossPanel  {
                background: #ffffff;
                border-top: 1px solid #e0e0e0;
                padding: 4px;
            }
            QPushButton#ProcessBtn {
                background-color: #1a73e8; color: white;
                border: none; border-radius: 5px;
                font-size: 13px; font-weight: bold; padding: 6px 12px;
            }
            QPushButton#ProcessBtn:hover    { background-color: #1558b0; }
            QPushButton#ProcessBtn:disabled { background-color: #b0bec5; }
            QPushButton#LearnBtn {
                background-color: #e65100; color: white;
                border: none; border-radius: 5px;
                font-size: 13px; font-weight: bold; padding: 6px 12px;
            }
            QPushButton#LearnBtn:hover    { background-color: #bf360c; }
            QPushButton#LearnBtn:disabled { background-color: #b0bec5; }
            QLineEdit {
                border: 1px solid #ccc; border-radius: 5px;
                padding: 6px 8px; font-size: 12px; background: #fafafa;
            }
            QLineEdit:focus { border: 1.5px solid #1a73e8; background: white; }
            QComboBox {
                border: 1px solid #ccc; border-radius: 4px;
                padding: 4px 8px; background: #fafafa; font-size: 11px;
            }
        """)
        # Disable buttons until startup finishes
        self._input_panel.set_enabled(False)

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e0e0e0;")
        line.setFixedHeight(1)
        return line

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("color: #e0e0e0;")
        line.setFixedWidth(1)
        return line
