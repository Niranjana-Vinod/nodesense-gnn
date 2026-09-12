#!/usr/bin/env bash
# NodeSense — One-command reproduction script (Linux, Mac, Colab)
set -e

echo "======================================================================"
echo " NodeSense — Explainable and Robust GNNs for Citation Networks"
echo "======================================================================"

# Install dependencies if not already installed
if ! python -c "import torch, torch_geometric, flask" >/dev/null 2>&1; then
    echo "[*] Installing dependencies..."
    pip install -r requirements.txt
fi

# Run training and evaluation pipeline
echo "[*] Running end-to-end training, evaluation, and ablations..."
python train_and_evaluate.py --epochs 100

# Start web application
echo "[*] Starting NodeSense Web Server..."
python backend/app.py
