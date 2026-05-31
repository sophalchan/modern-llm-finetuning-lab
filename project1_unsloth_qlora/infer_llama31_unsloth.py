#!/usr/bin/env python3
"""Run inference on Project 1 LoRA adapters (Unsloth)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import require_cuda


def infer(adapter_dir: str, prompt: str, max_new_tokens: int = 256) -> str:
    require_cuda()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = [{"role": "user", "content": prompt}]
    wrapped = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([wrapped], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="lora_model_llama3")
    p.add_argument("--prompt", default="Explain QLoRA in one paragraph for a security engineer.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()
    text = infer(args.adapter, args.prompt, args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
