#!/usr/bin/env python3
"""
Project 1 (8GB): Qwen2.5-1.5B + standard QLoRA instruction tuning.

Uses the same PyTorch / PEFT / TRL stack as Project 2 — no Unsloth required.
Public model — no HF_TOKEN needed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.utils import hf_token, load_yaml, require_cuda  # noqa: E402


def train(config_name: str = "project1_low_vram.yaml") -> None:
    require_cuda()
    cfg = load_yaml(config_name)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    token = hf_token()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_id = cfg["model_id"]
    max_seq_length = int(cfg["max_seq_length"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

    n = int(os.getenv("SFT_TRAIN_SAMPLES", cfg.get("train_samples", 200)))
    dataset = load_dataset(cfg["dataset_name"], split=f"train[:{n}]")

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
    training_args = SFTConfig(
        output_dir=str(ROOT / "outputs" / "project1"),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        warmup_steps=tcfg.get("warmup_steps", 2),
        max_steps=tcfg["max_steps"],
        learning_rate=tcfg["learning_rate"],
        logging_steps=tcfg.get("logging_steps", 2),
        max_length=max_seq_length,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
    )

    def formatting_func(examples):
        return examples["text"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"QLoRA training complete. Saved to {out}")


def main():
    p = argparse.ArgumentParser(description="Project 1: Qwen2.5-1.5B QLoRA (8GB)")
    p.add_argument("--config", default="project1_low_vram.yaml")
    args = p.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
