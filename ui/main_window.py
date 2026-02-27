"""
MainWindow — application shell and signal/slot orchestration hub (v3: Ollama).

v3 changes:
  • Encoder switched from SentenceTransformer to OllamaEncoder (llama3.2:3b).
  • LLMWorker added: runs ResponseGenerator.generate() off the main thread
    after each forward pass, displaying the LLM reply in the output panel.
  • StartupWorker loads OllamaEncoder instead of SemanticEncoder.
  • Header subtitle updated to reflect Ollama stack.
"""

from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui     import QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QSizePolicy,
    QProgressDialog, QApplication,
)

from config import PRETRAIN_EPOCHS, MEMORY_THRESHOLD, BRAIN_SAVE_PATH, CLASS_NAMES
from core.memory               import BrainMemory
from core.projection           import ActivationProjector
from visualization.brain_canvas import BrainCanvas
from ui.input_panel             import InputPanel
from ui.output_panel            import OutputPanel
from ui.loss_graph              import LossGraph
from ui.confidence_graph        import ConfidenceGraph
from explainability.analyzer    import PathAnalyzer


# ── Startup worker ────────────────────────────────────────────────────────────

class StartupWorker(QObject):
    """
    Runs off the main thread:
      1. Instantiates OllamaEncoder (connects to local Ollama server)
      2. Creates Trainer(encoder)
      3. Calls trainer.pretrain()
    Emits finished(encoder, trainer) when done.
    """
    progress = pyqtSignal(str, int)          # message, percentage 0..100
    finished = pyqtSignal(object, object)    # encoder, trainer
    error    = pyqtSignal(str)

    def run(self):
        try:
            # Step 1: connect to Ollama and probe embedding dimension
            self.progress.emit("Connecting to Ollama…", 5)
            from core.ollama_encoder import OllamaEncoder
            encoder = OllamaEncoder()
            self.progress.emit(
                f"Encoder ready  [{encoder.model_name}  dim={encoder.embedding_dim}]", 20
            )

            # Step 2: build trainer
            from core.trainer import Trainer
            trainer = Trainer(encoder)

            # Try loading saved brain first — skip pretraining if successful
            if trainer.load():
                self.progress.emit(
                    f"Saved brain loaded — {len(trainer.loss_history)} prior steps", 95
                )
            else:
                self.progress.emit("Pre-training deep network (300 epochs)…", 22)

                def cb(epoch: int, loss: float):
                    pct = 22 + int(epoch / PRETRAIN_EPOCHS * 76)
                    self.progress.emit(
                        f"Training epoch {epoch}/{PRETRAIN_EPOCHS}   loss={loss:.4f}", pct
                    )

                trainer.pretrain(progress_cb=cb)
                trainer.save()   # persist so next launch skips pretraining
                self.progress.emit("Brain trained and saved.", 100)

            self.finished.emit(encoder, trainer)

        except Exception as exc:
            self.error.emit(str(exc))


# ── LLM response worker ──────────────────────────────────────────────────────

class LLMWorker(QObject):
    """
    Runs ResponseGenerator.generate() off the main thread so the UI
    stays responsive while Ollama generates text.
    """
    response_ready = pyqtSignal(str)
    error          = pyqtSignal(str)

    # Slot — called via signal from main thread
    def generate(self, user_text: str, predicted_class: int, confidence: float):
        try:
            from core.response_generator import ResponseGenerator
            if not hasattr(self, "_gen"):
                self._gen = ResponseGenerator()
            reply = self._gen.generate(user_text, predicted_class, confidence)
            self.response_ready.emit(reply)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    # Internal signal to kick off LLM generation on the worker thread
    _request_llm = pyqtSignal(str, int, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Neural Network")
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
        self._last_embedding   = None
        self._last_visual_acts = None
        self._last_pred        = None
        self._last_probs       = None
        self._last_text        = None   # cached for LLM prompt

        # Projected weight cache — only recomputed after a learn step
        self._cached_visual_wm = None

        self._build_ui()
        self._connect_signals()
        self._setup_llm_worker()
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
        t = QLabel("Visual Neural Network")
        t.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        t.setObjectName("HeaderTitle")
        hrow.addWidget(t)
        hrow.addStretch()
        sub = QLabel(
            "Ollama · llama3.2:3b · 6-layer deep net · "
            "Associative Memory · Real-time PyTorch  |  F11 fullscreen"
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

        # Bottom strip: loss graph + confidence history side by side
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(0)

        self._loss_graph = LossGraph()
        self._loss_graph.setObjectName("LossPanel")
        bottom_row.addWidget(self._loss_graph, stretch=1)

        bottom_row.addWidget(self._vline())

        self._conf_graph = ConfidenceGraph()
        self._conf_graph.setObjectName("LossPanel")
        bottom_row.addWidget(self._conf_graph, stretch=1)

        outer.addLayout(bottom_row)

    # ── signal connections ────────────────────────────────────────────────────

    def _connect_signals(self):
        self._input_panel.process_requested.connect(self._on_process)
        self._input_panel.learn_requested.connect(self._on_learn)
        self._canvas.forward_animation_done.connect(self._on_forward_done)
        self._canvas.backprop_animation_done.connect(self._on_backprop_done)
        self._canvas.memory_recall_occurred.connect(self._on_memory_recall)

    # ── LLM worker setup ─────────────────────────────────────────────────────

    def _setup_llm_worker(self):
        self._llm_thread = QThread()
        self._llm_worker = LLMWorker()
        self._llm_worker.moveToThread(self._llm_thread)
        self._request_llm.connect(self._llm_worker.generate)
        self._llm_worker.response_ready.connect(self._on_llm_response)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_thread.start()

    # ── startup ───────────────────────────────────────────────────────────────

    def _start_startup(self):
        dlg = QProgressDialog(
            "Initialising Visual Neural Network…", None, 0, 100, self
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

        # Initialise canvas with projected weights and prime the cache
        wm                     = self._trainer.get_weight_magnitudes()
        self._cached_visual_wm = self._projector.project_all_weights(wm)
        self._canvas.update_weights(self._cached_visual_wm)

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

        # 1. Encode once — reused for memory store and infer (no double-encode)
        embedding = self._encoder.encode(text)
        self._last_embedding = embedding
        self._last_text      = text

        # 2. Memory recall (may return None bias)
        bias, similar, max_sim = self._memory.recall(embedding)

        # 3. Forward pass — pass pre-computed embedding, no internal re-encode
        probs, acts, pred, _ = self._trainer.infer(embedding, memory_bias=bias)

        # 4. Store in memory AFTER inference (reuse embedding, no re-encode)
        self._memory.store(embedding, text, pred, probs)

        # 5. Project activations; use cached weight projection (weights unchanged)
        visual_acts = self._projector.project_activations(acts)

        # 6. Clear old result while animation runs; show memory status
        self._output_panel.reset()
        self._input_panel.update_memory_status(
            self._memory.size(), self._memory.recent_texts(3)
        )
        self._input_panel.set_status("Propagating signal through network…")

        # 7. Trigger memory flash if recall happened
        if max_sim >= MEMORY_THRESHOLD:
            self._canvas.flash_memory_recall(max_sim)

        # 8. Start animation — use cached visual weights (no projection needed)
        self._canvas.animate_forward_pass(visual_acts, self._cached_visual_wm)

        # Cache for explainability
        self._last_visual_acts = visual_acts
        self._last_pred        = pred
        self._last_probs       = probs

    def _on_forward_done(self):
        # Animation complete — now reveal the prediction result
        if self._last_pred is not None and self._last_probs is not None:
            self._output_panel.set_prediction(self._last_pred, self._last_probs)
            self._input_panel.set_predicted_label(self._last_pred)
            self._conf_graph.add_inference(self._last_pred, float(max(self._last_probs)))

        if self._pending_learn is not None:
            self._on_learn_forward_done()
            return
        self._run_explainability()

        # Kick off LLM response generation (async, off main thread)
        if self._last_pred is not None and self._last_text is not None:
            confidence = float(max(self._last_probs)) if self._last_probs is not None else 0.0
            self._request_llm.emit(self._last_text, self._last_pred, confidence)

        self._set_busy(False)
        self._input_panel.set_status("Done. Try another sentence or press Learn.")

    # ── LLM response callbacks ────────────────────────────────────────────────

    def _on_llm_response(self, text: str):
        self._output_panel.set_llm_response(text)

    def _on_llm_error(self, msg: str):
        self._output_panel.set_llm_response(f"(LLM error: {msg})")

    # ── LEARN path ───────────────────────────────────────────────────────────

    def _on_learn(self, text: str, label: int):
        if self._busy or self._trainer is None:
            return
        self._set_busy(True)
        self._pending_learn = (text, label)
        self._output_panel.clear_memory_recall()

        # Encode once — reused for infer AND learn_from_input AND memory store
        embedding = self._encoder.encode(text)
        self._last_embedding = embedding
        self._last_text      = text

        bias, similar, max_sim = self._memory.recall(embedding)
        probs, acts, pred, _   = self._trainer.infer(embedding, memory_bias=bias)

        visual_acts = self._projector.project_activations(acts)

        self._output_panel.reset()
        self._input_panel.set_status("Forward pass… then learning…")

        if max_sim >= MEMORY_THRESHOLD:
            self._canvas.flash_memory_recall(max_sim)

        self._canvas.animate_forward_pass(visual_acts, self._cached_visual_wm)

        self._last_visual_acts = visual_acts
        self._last_pred        = pred
        self._last_probs       = probs

    def _on_learn_forward_done(self):
        text, label = self._pending_learn
        self._pending_learn = None

        # Use cached embedding — no re-encode needed
        embedding = self._last_embedding

        # Actual gradient step (pass pre-computed embedding)
        loss = self._trainer.learn_from_input(embedding, label)
        self._trainer.save()   # persist new knowledge immediately

        # Weights changed → update projection cache NOW (only time we recompute)
        new_wm                 = self._trainer.get_weight_magnitudes()
        self._cached_visual_wm = self._projector.project_all_weights(new_wm)

        self._loss_graph.add_loss(loss)
        self._output_panel.set_step_count(len(self._trainer.loss_history))
        self._input_panel.set_status(f"Backpropagating… loss={loss:.4f}")

        # Store the post-learn interaction (reuse cached embedding, no re-encode)
        self._memory.store(embedding, text, label,
                           self._last_probs if self._last_probs is not None
                           else [0.0] * 12)
        self._input_panel.update_memory_status(
            self._memory.size(), self._memory.recent_texts(3)
        )

        self._canvas.update_weights(self._cached_visual_wm)
        self._canvas.animate_backprop(self._trainer.weight_deltas)

    def _on_backprop_done(self):
        self._run_explainability()

        # Kick off LLM response for learn path too
        if self._last_pred is not None and self._last_text is not None:
            confidence = float(max(self._last_probs)) if self._last_probs is not None else 0.0
            self._request_llm.emit(self._last_text, self._last_pred, confidence)

        self._set_busy(False)
        self._input_panel.set_status("Network updated. Weights adjusted.")

    # ── memory recall banner ──────────────────────────────────────────────────

    def _on_memory_recall(self, similarity: float):
        self._output_panel.set_memory_recall(similarity)

    # ── explainability ────────────────────────────────────────────────────────

    def _run_explainability(self):
        if self._last_visual_acts is None or self._cached_visual_wm is None:
            return
        path, explanation = self._analyzer.analyse(
            self._last_visual_acts, self._cached_visual_wm, self._last_pred
        )
        self._canvas.highlight_neurons(path)
        self._output_panel.set_explanation(explanation)

    # ── keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool):
        self._busy = busy
        self._input_panel.set_enabled(not busy)
        if busy:
            # Clear old highlights at the START of a new inference,
            # not at the end — so explainability gold rings stay visible
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
