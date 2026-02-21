"""
Global configuration — Artificial Brain Simulator (v2: deep transformer stack).

All architecture constants, timing values, and colours live here.
One change here propagates to every module automatically.
"""

import torch

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Network Architecture ─────────────────────────────────────────────────────
# ACTUAL_LAYER_SIZES: real tensor dimensions inside DeepBrainNetwork
ACTUAL_LAYER_SIZES = [384, 256, 128, 64, 32, 12]

# VISUAL_LAYER_SIZES: neuron counts drawn in BrainCanvas (bucket-projected)
# Dense layers (384/256/128/64) each project down to 40 visual neurons.
# Smaller layers (32, 12) are rendered 1:1.
VISUAL_LAYER_SIZES = [40, 40, 40, 40, 32, 12]

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
    "Input\n(384→40)",
    "Sensory\n(256→40)",
    "Assoc.\n(128→40)",
    "Proc.\n(64→40)",
    "Decision\n(32)",
    "Intent\n(12)",
]

# ── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE   = 0.001      # lower LR for deeper network + BatchNorm
PRETRAIN_EPOCHS = 300        # 96 training samples × 12 classes

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

# ── Loss Graph ───────────────────────────────────────────────────────────────
LOSS_GRAPH_MAX_POINTS = 300
