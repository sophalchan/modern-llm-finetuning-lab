#!/usr/bin/env python3
"""Inference with Project 2 ORPO/DoRA adapters."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import hf_token, require_cuda


def infer(adapter_dir: str, prompt: str, max_new_tokens: int = 256) -> str:
    from src.inference_engine import OrpoInferenceEngine

    engine = OrpoInferenceEngine(adapter_dir)
    return engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="orpo_qwen_final")
    p.add_argument("--prompt", default="Write a safe, concise answer about incident response best practices.")
    args = p.parse_args()
    print(infer(args.adapter, args.prompt))


if __name__ == "__main__":
    main()
