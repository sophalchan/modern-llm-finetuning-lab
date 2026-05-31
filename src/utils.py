"""Shared helpers for modern LLM fine-tuning lab."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    path = ROOT / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for 8B/7B fine-tuning. "
            "Run: python scripts/check_environment.py"
        )


def hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
