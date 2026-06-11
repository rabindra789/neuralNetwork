# NeuroSight

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.4-41CD52?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)

**Visual Neural Network Simulator** — a real-time desktop application that visualises a deep transformer-based neural network as an interactive brain. Watch forward passes, backpropagation, short-term memory recall, and explainability highlights in real time.

---

## Features

- **6-layer neural network** visualisation (384→256→128→64→32→12) with live neuron activity
- **Forward pass animation** — travelling dots show signal flow through the network
- **Backpropagation visualisation** — orange pulse shows learning in reverse
- **12-class intent recognition** — trained on greeting, question, command, complaint, and more
- **Short-term memory** — associative recall with teal highlight animation
- **Explainability mode** — gold highlight traces the decision path
- **Real-time loss graph** — matplotlib live plot during pre-training
- **Semantic encoder** — sentence-transformers backend for text understanding
- **Interactive UI** — PyQt6 desktop interface with brain canvas and controls

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (optional, CPU fallback supported)

### Quick Install (WSL / Linux)

```bash
git clone https://github.com/defenx-tech/NeuroSight.git
cd NeuroSight
bash install.sh
```

### Manual Install

```bash
git clone https://github.com/defenx-tech/NeuroSight.git
cd NeuroSight
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install PyQt6 numpy matplotlib sentence-transformers
```

---

## Usage

```bash
python main.py
```

The application window opens with:

1. **Brain canvas** — 6-layer neural network visualisation with live neuron activity
2. **Input field** — type text and press Enter to see the forward pass animate
3. **Pre-training** — runs automatically on startup (300 epochs, ~30 seconds)
4. **Loss graph** — real-time plot of training loss

### Controls

- Type any sentence and watch the signal propagate through the network
- The network classifies into 12 intent categories
- Short-term memory recalls similar past inputs (teal highlight)
- Explainability mode shows the decision path (gold highlight)

---

## Architecture

```text
NeuroSight/
├── main.py                 # Application entry point
├── config.py               # Global configuration (architecture, timing, colors)
├── install.sh              # One-shot environment setup
├── requirements.txt        # Python dependencies
├── core/
│   ├── encoder.py          # Text encoder interface
│   ├── memory.py           # Short-term memory ring buffer
│   ├── network.py          # DeepBrainNetwork model
│   ├── projection.py       # Dimensionality projection
│   ├── semantic_encoder.py # sentence-transformers wrapper
│   └── trainer.py          # Training loop
├── ui/
│   └── main_window.py      # PyQt6 main window
├── visualization/
│   └── ...
├── explainability/
│   └── ...
└── ...
```

---

## Example

```python
# Programmatic usage (import core modules)
from core.network import DeepBrainNetwork
from core.semantic_encoder import SemanticEncoder
import torch

# Load the network
model = DeepBrainNetwork()
model.load_state_dict(torch.load("brain_weights.pt"))
model.eval()

# Encode text
enc = SemanticEncoder()
text = "Hello, how are you today?"
embedding = enc.encode(text)

# Run inference
with torch.no_grad():
    logits = model(torch.tensor(embedding).unsqueeze(0))
    probs = torch.softmax(logits, dim=1)
    intent_idx = probs.argmax().item()

print(f"Predicted intent: {config.CLASS_NAMES[intent_idx]}")
```

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — neural network engine
- **PyQt6** — desktop GUI framework
- **NumPy / Matplotlib** — data handling and plotting
- **sentence-transformers** — semantic text encoding
- **CUDA 12.8** — GPU acceleration (RTX 3050+)

---

## License

MIT License — see [LICENSE](LICENSE).
