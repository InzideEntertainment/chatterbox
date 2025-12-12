#!/bin/bash
set -e

echo "=== Chatterbox LXC GPU Bootstrap ==="

# ------------------------------------------------------------------------------
# 1) Persistent venv
# ------------------------------------------------------------------------------
echo "[1/5] Creating persistent venv..."
mkdir -p /var/lib/venv/chatterbox
python3 -m venv /var/lib/venv/chatterbox/venv --upgrade-deps
rm -rf /opt/chatterbox/venv 2>/dev/null || true
ln -s /var/lib/venv/chatterbox/venv /opt/chatterbox/venv
source /opt/chatterbox/venv/bin/activate

# ------------------------------------------------------------------------------
# 2) PyTorch + torchaudio (CUDA 12.4 wheels; drop +cu124 if using CPU-only/CUDA11)
# ------------------------------------------------------------------------------
echo "[2/5] Installing PyTorch/torchaudio..."
pip install --upgrade pip wheel setuptools
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------------------------
# 3) Install chatterbox in editable mode (run from repo root)
# ------------------------------------------------------------------------------
echo "[3/5] Installing chatterbox (editable)..."
pip install -e .

# ------------------------------------------------------------------------------
# 4) Validate CUDA
# ------------------------------------------------------------------------------
echo "[4/5] Validating CUDA..."
python3 - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
EOF

# ------------------------------------------------------------------------------
# 5) Done
# ------------------------------------------------------------------------------
echo "=== SETUP COMPLETE ==="
echo "Activate venv with: source /opt/chatterbox/venv/bin/activate"
echo "Run gradio with:    python3 gradio_tts_app.py"
