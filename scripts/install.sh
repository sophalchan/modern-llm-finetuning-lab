#!/usr/bin/env bash
# Install Modern LLM Fine-Tuning Lab
# Requires: Python 3.11, NVIDIA GPU, Linux
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3.11}"
if ! command -v "$PY" &>/dev/null; then
  echo "ERROR: python3.11 required (Python 3.14 breaks ML packages)."
  echo "Install: sudo apt install python3.11 python3.11-venv"
  exit 1
fi

echo "=== Installing with $($PY --version) ==="
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Pinned stack tested on RTX 2000 Ada (8GB VRAM)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.44,<4.50" "datasets>=2.19" "peft>=0.12" \
  "accelerate>=0.30" "bitsandbytes>=0.43" "trl>=0.11" \
  pyyaml python-dotenv rich sentencepiece loralib

cp -n .env.example .env 2>/dev/null || true

echo ""
python scripts/check_environment.py
echo ""
echo "=== Install complete ==="
echo ""
echo "Project 2 (ORPO + DoRA) — works on 8GB GPU:"
echo "  python project2_orpo_dora/train_qwen_orpo.py --config project2_low_vram.yaml"
echo "  python app_ui.py   # Web UI at http://localhost:7860"
echo ""
echo "Project 1 (Unsloth QLoRA) — needs 16GB+ VRAM + HF_TOKEN in .env:"
echo "  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
echo "  python project1_unsloth_qlora/train_llama31_unsloth.py --config project1_low_vram.yaml"
