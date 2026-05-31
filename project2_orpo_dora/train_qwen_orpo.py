#!/usr/bin/env python3
"""
Project 2: Qwen2.5-7B + ORPO with optional DoRA (single-step preference alignment).

Requires: Linux + NVIDIA GPU (16GB+ VRAM), CUDA, bitsandbytes.
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


def train(config_name: str = "project2_orpo_dora.yaml") -> None:
    require_cuda()
    cfg = load_yaml(config_name)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import ORPOConfig, ORPOTrainer

    token = hf_token()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_id = cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_dora=cfg.get("use_dora", True),
    )
    model = get_peft_model(model, peft_config)

    n = int(os.getenv("ORPO_TRAIN_SAMPLES", cfg.get("train_samples", 2000)))
    dataset = load_dataset(cfg["dataset_name"], split=f"train[:{n}]")

    def process_orpo_format(examples):
        return {
            "prompt": examples["question"],
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }

    dataset = dataset.map(process_orpo_format)
    dataset = dataset.train_test_split(test_size=cfg.get("test_size", 0.1))

    ocfg = cfg["orpo"]
    orpo_config = ORPOConfig(
        output_dir=str(ROOT / "outputs" / "project2"),
        learning_rate=ocfg["learning_rate"],
        per_device_train_batch_size=ocfg["per_device_train_batch_size"],
        gradient_accumulation_steps=ocfg["gradient_accumulation_steps"],
        max_prompt_length=ocfg["max_prompt_length"],
        max_length=ocfg["max_length"],
        beta=ocfg["beta"],
        logging_steps=ocfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=ocfg["eval_steps"],
        bf16=True,
        remove_unused_columns=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=orpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    try:
        trainer = ORPOTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = ORPOTrainer(**trainer_kwargs, tokenizer=tokenizer)

    trainer.train()
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"ORPO training complete. Saved to {out}")


def main():
    p = argparse.ArgumentParser(description="Project 2: Qwen2.5-7B ORPO + DoRA")
    p.add_argument("--config", default="project2_orpo_dora.yaml")
    p.add_argument("--no-dora", action="store_true", help="Use standard LoRA instead of DoRA")
    args = p.parse_args()
    if args.no_dora:
        cfg = load_yaml(args.config)
        cfg["use_dora"] = False
        import yaml
        tmp = ROOT / "configs" / "_project2_no_dora.yaml"
        tmp.write_text(yaml.dump(cfg), encoding="utf-8")
        train("_project2_no_dora.yaml")
    else:
        train(args.config)


if __name__ == "__main__":
    main()
