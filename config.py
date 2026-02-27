"""
Global configuration — Artificial Brain Simulator (v3: Ollama-powered).

All architecture constants, timing values, and colours live here.
One change here propagates to every module automatically.
"""

import torch

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Network Architecture ─────────────────────────────────────────────────────
# ACTUAL_LAYER_SIZES: real tensor dimensions inside DeepBrainNetwork
# Input 3072 comes from llama3.2:3b embeddings via Ollama
ACTUAL_LAYER_SIZES = [3072, 512, 256, 128, 64, 12]

# VISUAL_LAYER_SIZES: neuron counts drawn in BrainCanvas
# Set to actual sizes — the 3D canvas renders every real neuron.
VISUAL_LAYER_SIZES = [3072, 512, 256, 128, 64, 12]

# 12 intent classes
CLASS_NAMES = [
    "Greeting",
    "Farewell",
    "What-Q",
    "How-Q",
    "Help Req.",
    "Info Req.",
    "Confirm",
    "Praise",
    "Deny",
    "Complaint",
    "Command",
    "Small Talk",
]

# Labels printed under each layer column in the brain canvas
BRAIN_REGION_LABELS = [
    "Input\n(3072)",
    "Sensory\n(512)",
    "Assoc.\n(256)",
    "Proc.\n(128)",
    "Decision\n(64)",
    "Intent\n(12)",
]

# ── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE      = 0.001   # lower LR for deeper network + LayerNorm
PRETRAIN_EPOCHS    = 300     # 240 training samples × 12 classes
UNCERTAINTY_THRESH = 0.40    # max(probs) below this → show "Uncertain"
BRAIN_SAVE_PATH    = "brain_weights.pt"   # persisted network state

# ── Short-term Memory ────────────────────────────────────────────────────────
MEMORY_CAPACITY   = 10       # maximum stored interactions (ring buffer)
MEMORY_THRESHOLD  = 0.65     # cosine similarity threshold for recall trigger
MEMORY_BIAS_SCALE = 0.10     # scale of recalled-embedding nudge to input
MEMORY_FLASH_MS   = 700      # teal input-layer flash duration (ms)

# ── Animation Timing (milliseconds) ──────────────────────────────────────────
# Total forward animation = 6×160 + 5×200 = 1960 ms across 6 layers
LAYER_ANIM_DELAY     = 160   # pause between layer activate and pulse start
PULSE_DURATION_MS    = 200   # travelling-dot flight time per layer pair
BACKPROP_DURATION_MS = 800
FRAME_INTERVAL_MS    = 16    # ~60 fps

# ── Visual Constants ─────────────────────────────────────────────────────────
NEURON_RADIUS   = 14
CANVAS_MARGIN_X = 80
CANVAS_MARGIN_Y = 45

# ── Colours (R, G, B) ────────────────────────────────────────────────────────
COLOR_BG               = (255, 255, 255)
COLOR_NEURON_INACTIVE  = (245, 245, 245)
COLOR_NEURON_ACTIVE    = (30,  110, 220)   # blue
COLOR_NEURON_BORDER    = (30,   30,  30)
COLOR_CONN_WEAK        = (210, 210, 210)
COLOR_CONN_STRONG      = (90,   90,  90)
COLOR_PULSE            = (50,  160, 255)   # forward-pass dot (blue)
COLOR_BACKPROP         = (255, 140,   0)   # learning flash (orange)
COLOR_HIGHLIGHT        = (255, 200,  40)   # explainability path (gold)
COLOR_MEMORY           = (0,   200, 180)   # associative memory recall (teal)
COLOR_LABEL            = (80,   80,  80)

# ── 3D Canvas ───────────────────────────────────────────────────────────────
LAYER_SPACING_3D      = 12.0   # X-axis distance between consecutive layers
LAYER_RADIUS_3D       = 5.0    # YZ circle radius for dense layers (40 neurons)
OUTPUT_RADIUS_3D      = 2.5    # smaller circle for output layer (12 neurons)
NEURON_SIZE_3D        = 4.0    # scatter point size (px) — smaller for 4000+ neurons
NEURON_SIZE_HIGHLIGHT = 10.0   # highlighted neuron size (px)
MAX_CONNS_PER_PAIR    = 800    # subsample connections when src×dst exceeds this
CAM_DISTANCE_3D       = 85.0
CAM_ELEVATION_3D      = 25.0
CAM_AZIMUTH_3D        = -60.0

# ── 3D Dark Colour Scheme (RGBA float 0–1) ─────────────────────────────────
COLOR_3D_BG              = (0.06, 0.06, 0.10, 1.0)
COLOR_3D_NEURON_INACTIVE = (0.25, 0.25, 0.30, 0.7)
COLOR_3D_NEURON_ACTIVE   = (0.12, 0.43, 0.86, 1.0)
COLOR_3D_CONN_WEAK       = (0.20, 0.20, 0.25, 0.08)
COLOR_3D_CONN_STRONG     = (0.40, 0.45, 0.55, 0.35)
COLOR_3D_PULSE           = (0.20, 0.63, 1.00, 0.90)
COLOR_3D_BACKPROP        = (1.00, 0.55, 0.00, 0.85)
COLOR_3D_HIGHLIGHT       = (1.00, 0.78, 0.16, 1.0)
COLOR_3D_MEMORY          = (0.00, 0.78, 0.71, 0.90)
COLOR_3D_LABEL           = (0.70, 0.75, 0.85, 1.0)

# ── Loss Graph ───────────────────────────────────────────────────────────────
LOSS_GRAPH_MAX_POINTS = 300
