"""
BrainCanvas — the 'artificial brain' visualisation widget (v2).

v2 changes vs v1:
  • Uses VISUAL_LAYER_SIZES (6 layers: [40,40,40,40,32,12]) instead of LAYER_SIZES.
  • animate_forward_pass() is fully dynamic — scales to any number of layers.
  • Brain-region labels from BRAIN_REGION_LABELS (config).
  • Memory-recall teal flash: flash_memory_recall(intensity) lights the input
    layer in teal and decays over MEMORY_FLASH_MS milliseconds.
  • New signal: memory_recall_occurred(float).
"""

import random
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore    import Qt, QTimer, QRect, QPoint, pyqtSignal
from PyQt6.QtGui     import QPainter, QColor, QFont, QPen, QPixmap
from PyQt6.QtWidgets import QWidget, QToolTip

from config import (
    VISUAL_LAYER_SIZES, BRAIN_REGION_LABELS,
    CANVAS_MARGIN_X, CANVAS_MARGIN_Y, NEURON_RADIUS,
    LAYER_ANIM_DELAY, PULSE_DURATION_MS,
    BACKPROP_DURATION_MS, MEMORY_FLASH_MS,
    FRAME_INTERVAL_MS,
    COLOR_BG, COLOR_LABEL,
)
from visualization.neuron_item      import NeuronItem
from visualization.connection_item  import ConnectionItem


class BrainCanvas(QWidget):
    """
    Live neural-network visualisation panel.

    Signals
    -------
    forward_animation_done   — emitted when the last layer activates.
    backprop_animation_done  — emitted when the orange flash fully decays.
    memory_recall_occurred   — emitted with similarity score on teal flash.
    """

    forward_animation_done  = pyqtSignal()
    backprop_animation_done = pyqtSignal()
    memory_recall_occurred  = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(860, 520)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self._neurons:     List[List[NeuronItem]]     = []
        self._connections: List[List[ConnectionItem]] = []
        self._activations:       Optional[List[np.ndarray]] = None
        self._weight_magnitudes: Optional[List[np.ndarray]] = None

        # Cached pixmap of static connection lines — rebuilt only on layout/weight change
        self._conn_pixmap: Optional[QPixmap] = None

        # Forward-pass pulse
        self._pulse_layer_pair: Optional[int] = None
        self._pulse_start:      float         = 0.0
        self._pulse_duration:   float         = PULSE_DURATION_MS / 1000.0

        # Backprop flash
        self._backprop_active:   bool  = False
        self._backprop_start:    float = 0.0
        self._backprop_duration: float = BACKPROP_DURATION_MS / 1000.0

        # Memory recall teal flash (input layer only)
        self._memory_alpha:    float = 0.0
        self._memory_start:    float = 0.0
        self._memory_duration: float = MEMORY_FLASH_MS / 1000.0

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(FRAME_INTERVAL_MS)
        self._frame_timer.timeout.connect(self._tick)
        self._frame_timer.start()

        # Idle ambient animation — pulses a random layer every 500 ms when quiet
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(500)
        self._idle_timer.timeout.connect(self._idle_tick)
        self._idle_timer.start()

        self.setMouseTracking(True)

    # ── layout ───────────────────────────────────────────────────────────────

    def showEvent(self, event):
        self._build_items()
        super().showEvent(event)

    def resizeEvent(self, event):
        self._build_items()
        super().resizeEvent(event)

    def _build_items(self):
        """Compute pixel positions for every visual neuron and connection."""
        w, h = self.width(), self.height()
        if w < 100 or h < 100:
            return

        n_layers = len(VISUAL_LAYER_SIZES)
        avail_x  = w - 2 * CANVAS_MARGIN_X
        avail_y  = h - 2 * CANVAS_MARGIN_Y

        layer_xs = [
            CANVAS_MARGIN_X + avail_x * i / (n_layers - 1)
            for i in range(n_layers)
        ]

        self._neurons = []
        for layer_idx, (n_vis, lx) in enumerate(zip(VISUAL_LAYER_SIZES, layer_xs)):
            spacing = avail_y / max(n_vis, 2)
            radius  = max(5, min(NEURON_RADIUS, int(spacing * 0.36)))
            if n_vis == 1:
                ys = [h / 2.0]
            else:
                ys = [
                    CANVAS_MARGIN_Y + avail_y * i / (n_vis - 1)
                    for i in range(n_vis)
                ]
            layer_items = []
            for ni, ny in enumerate(ys):
                item        = NeuronItem(lx, ny, layer_idx, ni)
                item.radius = radius
                layer_items.append(item)
            self._neurons.append(layer_items)

        self._connections = []
        for lp in range(n_layers - 1):
            pair = [
                ConnectionItem(src, dst, lp)
                for src in self._neurons[lp]
                for dst in self._neurons[lp + 1]
            ]
            self._connections.append(pair)

        if self._weight_magnitudes is not None:
            self._apply_weights(self._weight_magnitudes)
        self._rebuild_conn_pixmap()
        self.update()

    # ── public API ────────────────────────────────────────────────────────────

    def update_weights(self, weight_magnitudes: List[np.ndarray]):
        self._weight_magnitudes = weight_magnitudes
        self._apply_weights(weight_magnitudes)
        self._rebuild_conn_pixmap()
        self.update()

    def animate_forward_pass(
        self,
        activations:       List[np.ndarray],
        weight_magnitudes: List[np.ndarray],
    ):
        """
        Dynamic forward-pass animation for any number of layers.

        Phase schedule (d = LAYER_ANIM_DELAY, p = PULSE_DURATION_MS):
          activate(i)  at  t = i*(d+p)   ms
          pulse(i)     at  t = i*(d+p)+d ms
          done         at  t = n*d + (n-1)*p ms
        """
        self._activations       = activations
        self._weight_magnitudes = weight_magnitudes
        self._apply_weights(weight_magnitudes)
        self._rebuild_conn_pixmap()
        for layer in self._neurons:
            for n in layer:
                n.reset()
        self.update()

        d        = LAYER_ANIM_DELAY
        p        = PULSE_DURATION_MS
        n_layers = len(VISUAL_LAYER_SIZES)

        phases = []
        for i in range(n_layers):
            phases.append((i * (d + p),     'activate', i))
            if i < n_layers - 1:
                phases.append((i * (d + p) + d, 'pulse',    i))
        phases.append((n_layers * d + (n_layers - 1) * p, 'done', None))

        for delay_ms, phase_type, arg in phases:
            if phase_type == 'activate':
                QTimer.singleShot(delay_ms,
                    lambda li=arg: self._phase_activate_layer(li))
            elif phase_type == 'pulse':
                QTimer.singleShot(delay_ms,
                    lambda li=arg: self._phase_start_pulse(li))
            else:
                QTimer.singleShot(delay_ms, self.forward_animation_done.emit)

    def animate_backprop(self, weight_deltas: Optional[List[np.ndarray]] = None):
        self._backprop_active = True
        self._backprop_start  = time.monotonic()
        self.update()

    def flash_memory_recall(self, similarity: float):
        """Teal input-layer flash indicating associative memory recall."""
        self._memory_alpha    = float(min(1.0, max(0.0, similarity)))
        self._memory_start    = time.monotonic()
        self.memory_recall_occurred.emit(similarity)
        self.update()

    def highlight_neurons(self, path: List[Tuple[int, int]]):
        self.clear_highlights()
        for (li, ni) in path:
            if li < len(self._neurons) and ni < len(self._neurons[li]):
                self._neurons[li][ni].highlighted = True
        self.update()

    def clear_highlights(self):
        for layer in self._neurons:
            for n in layer:
                n.highlighted = False
        self.update()

    # ── mouse interaction ─────────────────────────────────────────────────────

    def mouseMoveEvent(self, event):
        pos = event.position()
        mx, my = pos.x(), pos.y()
        for layer_idx, layer in enumerate(self._neurons):
            for neuron in layer:
                dx = mx - neuron.x
                dy = my - neuron.y
                if (dx * dx + dy * dy) <= (neuron.radius + 4) ** 2:
                    label    = BRAIN_REGION_LABELS[layer_idx].replace("\n", " ")
                    act_pct  = int(abs(neuron.activation) * 100)
                    tip = (
                        f"<b>{label}</b><br>"
                        f"Neuron #{neuron.neuron_idx}<br>"
                        f"Activation: {act_pct}%"
                    )
                    QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                    return
        QToolTip.hideText()

    # ── phase handlers ────────────────────────────────────────────────────────

    def _phase_activate_layer(self, layer_idx: int):
        if self._activations is None or layer_idx >= len(self._activations):
            return
        acts    = self._activations[layer_idx]
        max_act = float(np.max(np.abs(acts))) + 1e-8
        for ni, neuron in enumerate(self._neurons[layer_idx]):
            if ni < len(acts):
                neuron.set_activation(float(acts[ni]) / max_act)
        self.update()

    def _phase_start_pulse(self, layer_pair_idx: int):
        self._pulse_layer_pair = layer_pair_idx
        self._pulse_start      = time.monotonic()

    # ── 60 fps tick ──────────────────────────────────────────────────────────

    def _tick(self):
        now = time.monotonic()
        needs_paint = False

        if self._pulse_layer_pair is not None:
            if now - self._pulse_start >= self._pulse_duration:
                self._pulse_layer_pair = None
            needs_paint = True

        if self._backprop_active:
            if now - self._backprop_start >= self._backprop_duration:
                self._backprop_active = False
                self.backprop_animation_done.emit()
            needs_paint = True

        if self._memory_alpha > 0.0:
            elapsed = now - self._memory_start
            self._memory_alpha = float(max(0.0, 1.0 - elapsed / self._memory_duration))
            needs_paint = True

        if needs_paint:
            self.update()

    def _idle_tick(self):
        """
        Ambient pulse at ~2 fps — no decay in the 60 fps tick.

        Each fire: fade all neurons by 60%, then light one random layer.
        Runs entirely in Python but only 2× per second, so cost is negligible.
        """
        if (self._pulse_layer_pair is not None
                or self._backprop_active
                or not self._neurons):
            return
        # Multiplicative fade of all current activations (60% retained)
        for layer in self._neurons:
            for n in layer:
                n.activation *= 0.6
        # Light up a new random layer
        layer = random.choice(self._neurons)
        for neuron in layer:
            neuron.set_activation(random.uniform(0.10, 0.30))
        self.update()   # single repaint at 2 fps — not 60 fps

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._neurons:
            return

        now     = time.monotonic()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r, g, b = COLOR_BG
        painter.fillRect(self.rect(), QColor(r, g, b))

        pulse_t        = self._get_pulse_t(now)
        backprop_alpha = self._get_backprop_alpha(now)

        # Draw connections — fast path: blit cached pixmap + pulse dots only
        if self._conn_pixmap is not None and backprop_alpha == 0.0:
            painter.drawPixmap(0, 0, self._conn_pixmap)
            if self._pulse_layer_pair is not None and pulse_t is not None:
                for conn in self._connections[self._pulse_layer_pair]:
                    conn.paint_dot(painter, pulse_t)
        else:
            # Slow path: full per-connection render (only during 1 s backprop flash)
            for lp_idx, pair_conns in enumerate(self._connections):
                is_pulsing = (self._pulse_layer_pair == lp_idx)
                pt = pulse_t if is_pulsing else None
                for conn in pair_conns:
                    conn.paint(painter, pulse_t=pt, backprop_alpha=backprop_alpha)

        # Draw neurons — input layer gets memory_alpha; rest get 0.0
        for layer_idx, layer in enumerate(self._neurons):
            mem_a = self._memory_alpha if layer_idx == 0 else 0.0
            for neuron in layer:
                neuron.paint(
                    painter,
                    backprop_alpha=backprop_alpha,
                    memory_alpha=mem_a,
                )

        self._draw_labels(painter)
        painter.end()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_pulse_t(self, now: float) -> Optional[float]:
        if self._pulse_layer_pair is None:
            return None
        raw = (now - self._pulse_start) / self._pulse_duration
        raw = max(0.0, min(1.0, raw))
        return raw * raw * (3.0 - 2.0 * raw)   # smoothstep

    def _get_backprop_alpha(self, now: float) -> float:
        if not self._backprop_active:
            return 0.0
        return max(0.0, 1.0 - (now - self._backprop_start) / self._backprop_duration)

    def _apply_weights(self, weight_magnitudes: List[np.ndarray]):
        for lp_idx, pair_conns in enumerate(self._connections):
            if lp_idx >= len(weight_magnitudes):
                break
            w_mat  = weight_magnitudes[lp_idx]
            n_src  = VISUAL_LAYER_SIZES[lp_idx]
            n_dst  = VISUAL_LAYER_SIZES[lp_idx + 1]
            conn_i = 0
            for si in range(n_src):
                for di in range(n_dst):
                    if conn_i < len(pair_conns):
                        if di < w_mat.shape[0] and si < w_mat.shape[1]:
                            pair_conns[conn_i].weight_magnitude = float(w_mat[di, si])
                    conn_i += 1

    def _rebuild_conn_pixmap(self):
        """Render all connection lines to a QPixmap cache. Fast O(1) blit every frame."""
        w, h = self.width(), self.height()
        if w < 10 or h < 10 or not self._connections:
            self._conn_pixmap = None
            return
        pm = QPixmap(w, h)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for pair_conns in self._connections:
            for conn in pair_conns:
                conn.paint(p, pulse_t=None, backprop_alpha=0.0)
        p.end()
        self._conn_pixmap = pm

    def _draw_labels(self, painter: QPainter):
        lr, lg, lb = COLOR_LABEL
        painter.setPen(QPen(QColor(lr, lg, lb)))
        font = QFont("Segoe UI", 10)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        for layer_idx, layer_neurons in enumerate(self._neurons):
            if not layer_neurons or layer_idx >= len(BRAIN_REGION_LABELS):
                continue
            cx     = int(layer_neurons[0].x)
            bottom = max(int(n.y) for n in layer_neurons) + layer_neurons[0].radius + 6
            rect   = QRect(cx - 58, bottom, 116, 44)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                BRAIN_REGION_LABELS[layer_idx],
            )
