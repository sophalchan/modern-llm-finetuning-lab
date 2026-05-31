#!/usr/bin/env python3
"""Check GPU, CUDA, and optional package availability."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    print("=== Modern LLM Fine-Tuning Lab — Environment Check ===\n")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"VRAM: {vram:.1f} GB")
            if vram < 14:
                print("WARNING: <16GB VRAM — reduce batch size or use smaller models.")
        else:
            print("WARNING: No CUDA GPU — 8B/7B training will NOT run on this machine.")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    for pkg in ("transformers", "datasets", "peft", "trl", "bitsandbytes", "unsloth"):
        try:
            __import__(pkg)
            print(f"{pkg}: OK")
        except ImportError:
            print(f"{pkg}: missing")

    from src.utils import hf_token
    token = hf_token()
    print(f"HF_TOKEN set: {'yes' if token else 'no (needed for some gated models)'}")

    print("\nRecommended hardware:")
    print("  Project 1 QLoRA (Qwen 1.5B, train_qwen_qlora.py):  8 GB VRAM OK")
    print("  Project 1 Unsloth (Llama 8B):                       16–24 GB VRAM")
    print("  Project 2 ORPO/DoRA (Qwen 1.5B low_vram):           8 GB VRAM OK")
    print("  Project 2 ORPO/DoRA (Qwen 7B):                      16–24 GB VRAM")


if __name__ == "__main__":
    main()
