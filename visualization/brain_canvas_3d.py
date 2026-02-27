"""
BrainCanvas (3D) — OpenGL neural network visualization using pyqtgraph.

Drop-in replacement for the 2D BrainCanvas.  Same signals, same public API,
but renders the network as a 3D tunnel of circular neuron rings in a dark
environment with orbit camera controls.

Mouse controls (provided by GLViewWidget):
    Left-drag   : orbit (rotate around center)
    Right-drag  : pan
    Scroll      : zoom
"""

import random
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QToolTip

import pyqtgraph.opengl as gl

from config import (
    VISUAL_LAYER_SIZES, BRAIN_REGION_LABELS,
    LAYER_ANIM_DELAY, PULSE_DURATION_MS, BACKPROP_DURATION_MS,
    MEMORY_FLASH_MS, FRAME_INTERVAL_MS,
    LAYER_SPACING_3D, LAYER_RADIUS_3D, OUTPUT_RADIUS_3D,
    NEURON_SIZE_3D, NEURON_SIZE_HIGHLIGHT, MAX_CONNS_PER_PAIR,
    CAM_DISTANCE_3D, CAM_ELEVATION_3D, CAM_AZIMUTH_3D,
    COLOR_3D_BG, COLOR_3D_NEURON_INACTIVE, COLOR_3D_NEURON_ACTIVE,
    COLOR_3D_CONN_WEAK, COLOR_3D_CONN_STRONG,
    COLOR_3D_PULSE, COLOR_3D_BACKPROP, COLOR_3D_HIGHLIGHT,
    COLOR_3D_MEMORY, COLOR_3D_LABEL,
)


class _HoverGLView(gl.GLViewWidget):
    """GLViewWidget subclass that forwards hover events to the parent canvas."""

    def __init__(self, canvas, **kw):
        super().__init__(**kw)
        self._canvas = canvas

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if ev.buttons() == Qt.MouseButton.NoButton:
            self._canvas._handle_hover(ev)


class BrainCanvas(QWidget):
    """3D neural network visualization — API-compatible with the 2D version."""

    forward_animation_done  = pyqtSignal()
    backprop_animation_done = pyqtSignal()
    memory_recall_occurred  = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(860, 520)

        # ── GL view ──────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._gl = _HoverGLView(self)
        bg = COLOR_3D_BG
        self._gl.setBackgroundColor(
            int(bg[0] * 255), int(bg[1] * 255), int(bg[2] * 255)
        )
        self._gl.setCameraPosition(
            distance=CAM_DISTANCE_3D,
            elevation=CAM_ELEVATION_3D,
            azimuth=CAM_AZIMUTH_3D,
        )
        layout.addWidget(self._gl)

        # ── geometry ─────────────────────────────────────────────────────────
        self._layer_offsets: List[int] = []
        self._total_neurons = 0
        self._neuron_pos = np.empty((0, 3), dtype=np.float32)
        self._conn_verts = np.empty((0, 3), dtype=np.float32)
        self._conn_pair_idx = np.empty(0, dtype=np.int32)
        self._conn_flat = np.empty((0, 2), dtype=np.int32)
        self._n_conns = 0
        self._build_geometry()

        # ── state arrays ─────────────────────────────────────────────────────
        self._neuron_act = np.zeros(self._total_neurons, dtype=np.float32)
        self._neuron_hl  = np.zeros(self._total_neurons, dtype=bool)
        self._conn_weights = np.full(self._n_conns, 0.3, dtype=np.float32)

        # ── GL items ─────────────────────────────────────────────────────────
        self._scatter = gl.GLScatterPlotItem(
            pos=self._neuron_pos,
            color=self._make_neuron_colors(),
            size=NEURON_SIZE_3D,
            pxMode=True,
        )
        self._scatter.setGLOptions("translucent")
        self._gl.addItem(self._scatter)

        conn_colors = self._make_conn_colors()
        self._lines = gl.GLLinePlotItem(
            pos=self._conn_verts,
            color=conn_colors,
            width=1.0,
            mode="lines",
            antialias=True,
        )
        self._lines.setGLOptions("translucent")
        self._gl.addItem(self._lines)

        self._pulse_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(0, 0, 0, 0),
            size=6.0,
            pxMode=True,
        )
        self._pulse_scatter.setGLOptions("additive")
        self._gl.addItem(self._pulse_scatter)

        self._build_labels()

        # ── animation state ──────────────────────────────────────────────────
        self._activations: Optional[List[np.ndarray]] = None
        self._weight_magnitudes: Optional[List[np.ndarray]] = None

        self._pulse_layer_pair: Optional[int] = None
        self._pulse_start = 0.0
        self._pulse_duration = PULSE_DURATION_MS / 1000.0

        self._backprop_active = False
        self._backprop_start = 0.0
        self._backprop_duration = BACKPROP_DURATION_MS / 1000.0

        self._memory_alpha = 0.0
        self._memory_start = 0.0
        self._memory_duration = MEMORY_FLASH_MS / 1000.0

        # ── timers ───────────────────────────────────────────────────────────
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(FRAME_INTERVAL_MS)
        self._frame_timer.timeout.connect(self._tick)
        self._frame_timer.start()

        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(500)
        self._idle_timer.timeout.connect(self._idle_tick)
        self._idle_timer.start()

    # ── geometry construction ────────────────────────────────────────────────

    def _layer_radius(self, n_vis: int) -> float:
        """Scale radius by cube-root of neuron count for uniform density."""
        max_n = max(VISUAL_LAYER_SIZES)
        return max(OUTPUT_RADIUS_3D,
                   LAYER_RADIUS_3D * (n_vis / max_n) ** (1.0 / 3.0))

    def _build_geometry(self):
        """Compute static 3D positions — structured input/output, random hidden."""
        rng = np.random.RandomState(42)  # deterministic layout across runs
        positions = []
        offsets = []
        offset = 0
        n_layers = len(VISUAL_LAYER_SIZES)
        total_x = (n_layers - 1) * LAYER_SPACING_3D
        x_start = -total_x / 2.0
        last_layer = n_layers - 1
        golden = (1.0 + np.sqrt(5.0)) / 2.0  # for Fibonacci sphere

        self._layer_radii: List[float] = []

        for li, n_vis in enumerate(VISUAL_LAYER_SIZES):
            offsets.append(offset)
            x_center = x_start + li * LAYER_SPACING_3D
            radius = self._layer_radius(n_vis)
            self._layer_radii.append(radius)

            if li == 0:
                # ── Input layer: Fibonacci sphere (structured, evenly spaced) ──
                x_spread = radius * 0.30
                for ni in range(n_vis):
                    theta = 2.0 * np.pi * ni / golden
                    phi = np.arccos(1.0 - 2.0 * (ni + 0.5) / n_vis)
                    positions.append([
                        x_center + x_spread * np.cos(phi),
                        radius * np.sin(phi) * np.cos(theta),
                        radius * np.sin(phi) * np.sin(theta),
                    ])
            elif li == last_layer:
                # ── Output layer: circle ring ──
                for ni in range(n_vis):
                    theta = 2.0 * np.pi * ni / n_vis
                    positions.append([
                        x_center,
                        radius * np.cos(theta),
                        radius * np.sin(theta),
                    ])
            else:
                # ── Hidden layers: organic random scatter ──
                x_spread = LAYER_SPACING_3D * 0.20
                for ni in range(n_vis):
                    direction = rng.randn(3)
                    direction /= (np.linalg.norm(direction) + 1e-8)
                    r = rng.uniform(0.2, 1.0) ** (1.0 / 3.0)
                    positions.append([
                        x_center + direction[0] * x_spread * r,
                        direction[1] * radius * r,
                        direction[2] * radius * r,
                    ])
            offset += n_vis

        self._layer_offsets = offsets
        self._total_neurons = offset
        self._neuron_pos = np.array(positions, dtype=np.float32)

        # ── connections — subsample large layer pairs ─────────────────────────
        verts = []
        pair_idx = []
        flat_idx = []
        local_src = []
        local_dst = []

        for lp in range(n_layers - 1):
            n_src = VISUAL_LAYER_SIZES[lp]
            n_dst = VISUAL_LAYER_SIZES[lp + 1]
            total_conns = n_src * n_dst

            if total_conns <= MAX_CONNS_PER_PAIR:
                # draw every connection
                pairs = np.array(
                    [[si, di] for si in range(n_src) for di in range(n_dst)],
                    dtype=np.int32,
                )
            else:
                # randomly subsample
                chosen = rng.choice(total_conns, MAX_CONNS_PER_PAIR, replace=False)
                chosen.sort()
                pairs = np.column_stack([chosen // n_dst, chosen % n_dst]).astype(np.int32)

            for si, di in pairs:
                src_flat = offsets[lp] + si
                dst_flat = offsets[lp + 1] + di
                verts.append(self._neuron_pos[src_flat])
                verts.append(self._neuron_pos[dst_flat])
                pair_idx.append(lp)
                flat_idx.append([src_flat, dst_flat])
                local_src.append(si)
                local_dst.append(di)

        self._n_conns = len(pair_idx)
        if self._n_conns > 0:
            self._conn_verts = np.array(verts, dtype=np.float32)
            self._conn_pair_idx = np.array(pair_idx, dtype=np.int32)
            self._conn_flat = np.array(flat_idx, dtype=np.int32)
            self._conn_local_src = np.array(local_src, dtype=np.int32)
            self._conn_local_dst = np.array(local_dst, dtype=np.int32)
        else:
            self._conn_verts = np.zeros((0, 3), dtype=np.float32)
            self._conn_pair_idx = np.zeros(0, dtype=np.int32)
            self._conn_flat = np.zeros((0, 2), dtype=np.int32)
            self._conn_local_src = np.zeros(0, dtype=np.int32)
            self._conn_local_dst = np.zeros(0, dtype=np.int32)

    def _build_labels(self):
        """Add text labels beneath each layer cluster."""
        n_layers = len(VISUAL_LAYER_SIZES)
        total_x = (n_layers - 1) * LAYER_SPACING_3D
        x_start = -total_x / 2.0
        r, g, b, a = COLOR_3D_LABEL
        color_qt = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

        for li in range(n_layers):
            x = x_start + li * LAYER_SPACING_3D
            radius = self._layer_radii[li]
            label = BRAIN_REGION_LABELS[li].replace("\n", " ") if li < len(BRAIN_REGION_LABELS) else ""
            txt = gl.GLTextItem(
                pos=np.array([x, 0, -(radius + 2.2)]),
                text=label,
                color=color_qt,
                font=QFont("Segoe UI", 10),
            )
            self._gl.addItem(txt)

    # ── color computation (vectorized) ───────────────────────────────────────

    def _make_neuron_colors(self, backprop_alpha=0.0, memory_alpha=0.0):
        """Build (N, 4) RGBA float array for all neurons."""
        inactive = np.array(COLOR_3D_NEURON_INACTIVE, dtype=np.float32)
        active   = np.array(COLOR_3D_NEURON_ACTIVE,   dtype=np.float32)

        t = self._neuron_act[:, np.newaxis]  # (N, 1)
        colors = inactive[np.newaxis] * (1.0 - t) + active[np.newaxis] * t  # (N, 4)

        if backprop_alpha > 0.0:
            bp = np.array(COLOR_3D_BACKPROP, dtype=np.float32)
            colors = colors * (1.0 - backprop_alpha) + bp * backprop_alpha

        if memory_alpha > 0.0:
            mem = np.array(COLOR_3D_MEMORY, dtype=np.float32)
            start = self._layer_offsets[0]
            end = self._layer_offsets[1] if len(self._layer_offsets) > 1 else self._total_neurons
            colors[start:end] = (
                colors[start:end] * (1.0 - memory_alpha) + mem * memory_alpha
            )

        # highlights override
        hl_mask = self._neuron_hl
        if np.any(hl_mask):
            hl_color = np.array(COLOR_3D_HIGHLIGHT, dtype=np.float32)
            colors[hl_mask] = hl_color

        return np.clip(colors, 0.0, 1.0)

    def _make_neuron_sizes(self, memory_alpha=0.0):
        """Build (N,) size array."""
        sizes = np.full(self._total_neurons, NEURON_SIZE_3D, dtype=np.float32)
        if np.any(self._neuron_hl):
            sizes[self._neuron_hl] = NEURON_SIZE_HIGHLIGHT
        if memory_alpha > 0.0:
            start = self._layer_offsets[0]
            end = self._layer_offsets[1] if len(self._layer_offsets) > 1 else self._total_neurons
            sizes[start:end] += 6.0 * memory_alpha
        return sizes

    def _make_conn_colors(self, backprop_alpha=0.0):
        """Build (2*N_conn, 4) RGBA float array for line vertices."""
        if self._n_conns == 0:
            return np.zeros((0, 4), dtype=np.float32)

        weak   = np.array(COLOR_3D_CONN_WEAK,   dtype=np.float32)
        strong = np.array(COLOR_3D_CONN_STRONG,  dtype=np.float32)
        w = self._conn_weights[:, np.newaxis]  # (C, 1)
        base = weak[np.newaxis] * (1.0 - w) + strong[np.newaxis] * w  # (C, 4)

        if backprop_alpha > 0.0:
            bp = np.array(COLOR_3D_BACKPROP, dtype=np.float32)
            base = base * (1.0 - backprop_alpha) + bp * backprop_alpha

        # duplicate for both vertices of each segment
        colors = np.empty((self._n_conns * 2, 4), dtype=np.float32)
        colors[0::2] = base
        colors[1::2] = base
        return np.clip(colors, 0.0, 1.0)

    # ── refresh GL items ─────────────────────────────────────────────────────

    def _refresh_neurons(self, backprop_alpha=0.0, memory_alpha=0.0):
        colors = self._make_neuron_colors(backprop_alpha, memory_alpha)
        sizes  = self._make_neuron_sizes(memory_alpha)
        self._scatter.setData(color=colors, size=sizes)

    def _refresh_connections(self, backprop_alpha=0.0):
        colors = self._make_conn_colors(backprop_alpha)
        self._lines.setData(color=colors)

    # ── public API (same signatures as 2D BrainCanvas) ───────────────────────

    def update_weights(self, weight_magnitudes: List[np.ndarray]):
        """Store projected weight matrices and refresh connection visuals."""
        self._weight_magnitudes = weight_magnitudes
        for lp, wm in enumerate(weight_magnitudes):
            mask = self._conn_pair_idx == lp
            if not np.any(mask):
                continue
            srcs = self._conn_local_src[mask]
            dsts = self._conn_local_dst[mask]
            valid = (dsts < wm.shape[0]) & (srcs < wm.shape[1])
            weights = np.full(mask.sum(), 0.3, dtype=np.float32)
            if np.any(valid):
                weights[valid] = wm[dsts[valid], srcs[valid]].astype(np.float32)
            self._conn_weights[mask] = weights
        self._refresh_connections()

    def animate_forward_pass(
        self,
        activations: List[np.ndarray],
        weight_magnitudes: List[np.ndarray],
    ):
        """Orchestrate the forward-pass animation across all layers."""
        self._activations = activations
        self.update_weights(weight_magnitudes)

        # reset all neurons
        self._neuron_act[:] = 0.0
        self._refresh_neurons()

        d = LAYER_ANIM_DELAY
        p = PULSE_DURATION_MS
        n_layers = len(VISUAL_LAYER_SIZES)

        for i in range(n_layers):
            delay = i * (d + p)
            QTimer.singleShot(delay, lambda li=i: self._phase_activate_layer(li))
            if i < n_layers - 1:
                QTimer.singleShot(delay + d, lambda li=i: self._phase_start_pulse(li))

        total = n_layers * d + (n_layers - 1) * p
        QTimer.singleShot(total, self.forward_animation_done.emit)

    def animate_backprop(self, weight_deltas=None):
        """Trigger orange backprop flash overlay (800 ms decay)."""
        self._backprop_active = True
        self._backprop_start = time.monotonic()

    def flash_memory_recall(self, similarity: float):
        """Trigger teal flash on the input layer ring."""
        self._memory_alpha = float(min(1.0, max(0.0, similarity)))
        self._memory_start = time.monotonic()
        self.memory_recall_occurred.emit(similarity)

    def highlight_neurons(self, path: List[Tuple[int, int]]):
        """Set gold highlight on specific (layer_idx, neuron_idx) neurons."""
        self.clear_highlights()
        for li, ni in path:
            if li < len(self._layer_offsets):
                flat = self._layer_offsets[li] + ni
                if 0 <= flat < self._total_neurons:
                    self._neuron_hl[flat] = True
        self._refresh_neurons()

    def clear_highlights(self):
        """Remove all explainability highlights."""
        self._neuron_hl[:] = False
        self._refresh_neurons()

    # ── animation phases ─────────────────────────────────────────────────────

    def _phase_activate_layer(self, layer_idx: int):
        if self._activations is None or layer_idx >= len(self._activations):
            return
        acts = self._activations[layer_idx]
        max_act = float(np.max(np.abs(acts))) + 1e-8
        start = self._layer_offsets[layer_idx]
        n = min(VISUAL_LAYER_SIZES[layer_idx], len(acts))
        self._neuron_act[start:start + n] = np.maximum(0.0, acts[:n] / max_act)
        self._refresh_neurons(memory_alpha=self._memory_alpha)

    def _phase_start_pulse(self, layer_pair_idx: int):
        self._pulse_layer_pair = layer_pair_idx
        self._pulse_start = time.monotonic()

    # ── 60fps tick ───────────────────────────────────────────────────────────

    def _tick(self):
        now = time.monotonic()
        dirty = False

        # pulse dots
        if self._pulse_layer_pair is not None:
            elapsed = now - self._pulse_start
            if elapsed >= self._pulse_duration:
                self._pulse_layer_pair = None
                self._pulse_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
                dirty = True
            else:
                raw = elapsed / self._pulse_duration
                t = raw * raw * (3.0 - 2.0 * raw)  # smoothstep
                self._update_pulse_dots(t)
                dirty = True

        # backprop flash
        if self._backprop_active:
            elapsed = now - self._backprop_start
            if elapsed >= self._backprop_duration:
                self._backprop_active = False
                self._refresh_neurons()
                self._refresh_connections()
                self.backprop_animation_done.emit()
            else:
                alpha = max(0.0, 1.0 - elapsed / self._backprop_duration)
                self._refresh_neurons(backprop_alpha=alpha)
                self._refresh_connections(backprop_alpha=alpha)
            dirty = True

        # memory flash decay
        if self._memory_alpha > 0.0:
            elapsed = now - self._memory_start
            self._memory_alpha = float(max(0.0, 1.0 - elapsed / self._memory_duration))
            self._refresh_neurons(memory_alpha=self._memory_alpha)
            dirty = True

    def _idle_tick(self):
        """Ambient pulse — fade existing + random layer glow."""
        if self._pulse_layer_pair is not None or self._backprop_active:
            return
        self._neuron_act *= 0.6
        layer = random.randint(0, len(VISUAL_LAYER_SIZES) - 1)
        start = self._layer_offsets[layer]
        n = VISUAL_LAYER_SIZES[layer]
        self._neuron_act[start:start + n] = np.random.uniform(0.10, 0.30, size=n).astype(np.float32)
        self._refresh_neurons()

    # ── pulse dots ───────────────────────────────────────────────────────────

    def _update_pulse_dots(self, t: float):
        """Compute interpolated positions for travelling pulse dots."""
        lp = self._pulse_layer_pair
        if lp is None:
            return
        mask = self._conn_pair_idx == lp
        if not np.any(mask):
            return
        indices = self._conn_flat[mask]  # (K, 2)
        src_pos = self._neuron_pos[indices[:, 0]]
        dst_pos = self._neuron_pos[indices[:, 1]]
        dot_pos = src_pos + (dst_pos - src_pos) * t

        n = len(dot_pos)
        pulse_col = np.array(COLOR_3D_PULSE, dtype=np.float32)
        colors = np.tile(pulse_col, (n, 1))
        self._pulse_scatter.setData(pos=dot_pos, color=colors, size=6.0)

    # ── hover tooltips ───────────────────────────────────────────────────────

    def _handle_hover(self, ev):
        """Find nearest neuron to cursor via 3D→2D projection."""
        try:
            # Get the view and projection matrices
            view = self._gl.viewMatrix()
            proj = self._gl.projectionMatrix()
            vp = proj * view

            w = self._gl.width()
            h = self._gl.height()
            if w == 0 or h == 0:
                return

            # Project all neurons to screen space
            pos4 = np.ones((self._total_neurons, 4), dtype=np.float64)
            pos4[:, :3] = self._neuron_pos

            # Convert QMatrix4x4 to numpy 4x4
            m = np.array([vp.row(i).toTuple() for i in range(4)], dtype=np.float64)
            clip = (m @ pos4.T).T  # (N, 4)

            # perspective divide
            w_clip = clip[:, 3]
            valid = np.abs(w_clip) > 1e-6
            ndc = np.zeros((self._total_neurons, 2), dtype=np.float64)
            ndc[valid, 0] = clip[valid, 0] / w_clip[valid]
            ndc[valid, 1] = clip[valid, 1] / w_clip[valid]

            # NDC → screen
            sx = (ndc[:, 0] * 0.5 + 0.5) * w
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h

            pos = ev.position()
            mx, my = pos.x(), pos.y()
            dist = (sx - mx) ** 2 + (sy - my) ** 2

            nearest = int(np.argmin(dist))
            if dist[nearest] < 400:  # within ~20px
                # find layer/neuron indices
                li = 0
                for i, off in enumerate(self._layer_offsets):
                    if nearest >= off:
                        li = i
                label = BRAIN_REGION_LABELS[li].replace("\n", " ") if li < len(BRAIN_REGION_LABELS) else ""
                ni = nearest - self._layer_offsets[li]
                act_pct = int(abs(self._neuron_act[nearest]) * 100)
                tip = f"<b>{label}</b><br>Neuron #{ni}<br>Activation: {act_pct}%"
                QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
            else:
                QToolTip.hideText()
        except Exception:
            pass  # don't crash on tooltip errors
