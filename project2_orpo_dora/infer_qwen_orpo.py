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
    require_cuda()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = hf_token()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    base_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_id, token=token)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb, device_map="auto", token=token
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="orpo_qwen_final")
    p.add_argument("--prompt", default="Write a safe, concise answer about incident response best practices.")
    args = p.parse_args()
    print(infer(args.adapter, args.prompt))


if __name__ == "__main__":
    main()
