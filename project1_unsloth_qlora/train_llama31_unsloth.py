#!/usr/bin/env python3
"""
Project 1: Llama-3.1-8B + Unsloth QLoRA (fast domain adaptation).

Requires: Linux + NVIDIA GPU (16GB+ VRAM recommended), CUDA, Unsloth.
Set HF_TOKEN in .env for gated Llama weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.utils import load_yaml, require_cuda  # noqa: E402


def train(config_name: str = "project1_unsloth_qlora.yaml") -> None:
    require_cuda()
    cfg = load_yaml(config_name)

    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Unsloth not installed. Run:\n"
            '  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
            "  pip install --no-deps trl peft loralib sentencepiece"
        ) from exc

    max_seq_length = cfg["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=cfg.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset(cfg["dataset_name"], split="train")

    def format_prompts(examples):
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            user = f"{instruction}\n{input_text}".strip()
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": output},
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    tcfg = cfg["training"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=tcfg["per_device_train_batch_size"],
            gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
            warmup_steps=tcfg["warmup_steps"],
            max_steps=tcfg["max_steps"],
            learning_rate=tcfg["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=tcfg.get("logging_steps", 1),
            output_dir=str(ROOT / "outputs" / "project1"),
        ),
    )

    stats = trainer.train()
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Training complete. Adapters saved to {out}")
    print(stats)


def main():
    p = argparse.ArgumentParser(description="Project 1: Llama-3.1-8B Unsloth QLoRA")
    p.add_argument("--config", default="project1_unsloth_qlora.yaml")
    args = p.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
