# Modern LLM Fine-Tuning Lab

Two **local GPU projects** for efficient fine-tuning on consumer hardware (16–24 GB VRAM):

| Project | Model | Method | Goal |
|---------|-------|--------|------|
| **1 — Speed Run** | Llama-3.1-8B-Instruct | **QLoRA + Unsloth** | Fast domain / format adaptation |
| **2 — Preference Align** | Qwen2.5-7B-Instruct | **ORPO + DoRA** | Single-step style & preference tuning |

## Four techniques covered

1. **QLoRA** — 4-bit base model + LoRA adapters (low VRAM)  
2. **Unsloth** — optimized kernels (2–5× faster training)  
3. **ORPO** — one-step preference alignment (no separate SFT + DPO)  
4. **DoRA** — weight-decomposed LoRA (closer to full fine-tune quality)  

## Hardware requirements

- **Linux + NVIDIA GPU** with **16 GB+ VRAM** (24 GB recommended)  
- **CUDA** + PyTorch  
- **HF_TOKEN** in `.env` for gated models (Llama 3.1)  

This repo was built and verified on structure/scripts; **full 8B training must run on a CUDA machine**.

## Quick start

```bash
git clone https://github.com/sophalchan/modern-llm-finetuning-lab.git
cd modern-llm-finetuning-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-base.txt
cp .env.example .env   # add HF_TOKEN
python scripts/check_environment.py
```

### Project 1 — Llama + Unsloth QLoRA

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft loralib sentencepiece

python project1_unsloth_qlora/train_llama31_unsloth.py
python project1_unsloth_qlora/infer_llama31_unsloth.py --prompt "Explain SIEM correlation rules briefly."
python project1_unsloth_qlora/export_gguf.py   # Ollama / LM Studio
```

### Project 2 — Qwen + ORPO + DoRA

```bash
python project2_orpo_dora/train_qwen_orpo.py
python project2_orpo_dora/infer_qwen_orpo.py --prompt "Give a safe incident response summary."
python project2_orpo_dora/train_qwen_orpo.py --no-dora   # standard LoRA only
```

## Layout

```
configs/                    YAML hyperparameters
project1_unsloth_qlora/     Llama-3.1 + Unsloth QLoRA train/infer/export
project2_orpo_dora/         Qwen2.5 + ORPO/DoRA train/infer
data/sample_domain_instructions.jsonl
scripts/check_environment.py
```

**Author:** Sophal Chan · [Portfolio](https://sophalchan.github.io/)
