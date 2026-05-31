"""Shared PEFT adapter inference engine for CLI and Gradio UI."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_engines: dict[str, "PeftInferenceEngine"] = {}


class PeftInferenceEngine:
    def __init__(self, adapter_dir: str | Path):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        from .utils import hf_token

        adapter_path = Path(adapter_dir)
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_dir
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No trained adapters at {adapter_path}.\n"
                "Project 1: python project1_unsloth_qlora/train_qwen_qlora.py --config project1_low_vram.yaml\n"
                "Project 2: python project2_orpo_dora/train_qwen_orpo.py --config project2_low_vram.yaml"
            )

        cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
        self.base_id = cfg["base_model_name_or_path"]
        self.adapter_path = adapter_path
        self.use_dora = cfg.get("use_dora", False)
        self._adapter_name = adapter_path.name

        token = hf_token()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_id, token=token)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_id,
            quantization_config=bnb,
            device_map="auto",
            token=token,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        import torch

        messages = [{"role": "user", "content": prompt.strip()}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "assistant" in full.lower():
            parts = full.split("assistant")
            return parts[-1].strip() if len(parts) > 1 else full.strip()
        return full.strip()

    def info(self) -> dict:
        name = self._adapter_name.lower()
        if self.use_dora or "orpo" in name:
            method = "ORPO + DoRA" if self.use_dora else "ORPO + LoRA"
        else:
            method = "QLoRA"
        return {
            "base_model": self.base_id,
            "adapter_path": str(self.adapter_path),
            "method": method,
        }


# Backward-compatible alias used by Project 2 scripts
OrpoInferenceEngine = PeftInferenceEngine


def get_engine(adapter_dir: str = "orpo_qwen_final") -> PeftInferenceEngine:
    if adapter_dir not in _engines:
        _engines[adapter_dir] = PeftInferenceEngine(adapter_dir)
    return _engines[adapter_dir]
