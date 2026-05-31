#!/usr/bin/env python3
"""Export merged GGUF for Ollama / LM Studio (Unsloth)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import require_cuda


def export_gguf(adapter_dir: str, out_name: str = "merged_model", method: str = "q4_k_m") -> None:
    require_cuda()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    out = ROOT / out_name
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=method)
    print(f"GGUF export written under {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="lora_model_llama3")
    p.add_argument("--out", default="merged_model")
    p.add_argument("--method", default="q4_k_m")
    args = p.parse_args()
    export_gguf(args.adapter, args.out, args.method)


if __name__ == "__main__":
    main()
