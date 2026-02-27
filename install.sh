#!/usr/bin/env bash
# install.sh — one-shot environment setup for WSL (Ubuntu / Kali / Debian)
# Run once:  bash install.sh
# Then run:  python3 main.py

set -e

echo "=== Artificial Brain Simulator — Environment Setup ==="

# Detect if pip requires --break-system-packages (Kali / Debian Bookworm+)
PIP_FLAGS=""
if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
    python3 -m pip install --dry-run pip 2>&1 | grep -q "externally-managed" && \
        PIP_FLAGS="--break-system-packages"
fi

# PyTorch with CUDA 12.8 support (matches RTX 3050 / driver CUDA 13.1)
# NOTE: we do NOT upgrade pip — on Kali/Debian it is system-managed and
#       the upgrade would fail. The installed pip version is sufficient.
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    --quiet --root-user-action=ignore $PIP_FLAGS

# Remaining dependencies
python3 -m pip install PyQt6 numpy matplotlib \
    --quiet --root-user-action=ignore $PIP_FLAGS

# Ollama Python client
python3 -m pip install ollama \
    --quiet --root-user-action=ignore $PIP_FLAGS

echo ""
echo "=== Installation complete ==="
echo "Start the application with:"
echo "   cd $(dirname "$0") && python3 main.py"
