#!/usr/bin/env python3
"""Gradio web UI for Project 1 QLoRA fine-tuned model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr

from src.inference_engine import get_engine
from src.utils import require_cuda

ADAPTER_DIR = "lora_model_qwen"
PORT = 7861

EXAMPLES = [
    "Explain QLoRA in plain language for a beginner.",
    "What is fine-tuning and why use LoRA adapters?",
    "Give three tips for learning Python as a data science student.",
    "Summarize how a RAG system works in simple terms.",
]


def chat(message: str, max_tokens: int, temperature: float) -> str:
    if not message.strip():
        return "Please enter a question."
    try:
        engine = get_engine(ADAPTER_DIR)
        return engine.generate(message, max_new_tokens=int(max_tokens), temperature=float(temperature))
    except Exception as exc:
        return f"Error: {exc}"


def build_ui() -> gr.Blocks:
    try:
        require_cuda()
        meta = get_engine(ADAPTER_DIR).info()
        status = f"Ready — {meta['base_model']} ({meta['method']})"
    except Exception as exc:
        status = f"Model not loaded: {exc}"

    with gr.Blocks(title="QLoRA Fine-Tuned Assistant") as demo:
        gr.Markdown(
            """
            # Modern LLM Fine-Tuning Lab — Project 1 Web UI
            **Project 1:** Qwen + **QLoRA** instruction-tuned assistant (local GPU, 8 GB friendly)

            Ask general questions about AI/ML, programming, data science, or everyday topics.
            """
        )
        gr.Markdown(f"**Status:** {status}")

        with gr.Row():
            max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature")

        chatbot = gr.Chatbot(height=420, label="Chat")
        msg = gr.Textbox(label="Your question", placeholder="Type a question and press Send…")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        gr.Examples(examples=EXAMPLES, inputs=msg)

        def respond(user_message, chat_history, mt, temp):
            chat_history = chat_history or []
            reply = chat(user_message, mt, temp)
            chat_history = chat_history + [(user_message, reply)]
            return "", chat_history

        send.click(respond, [msg, chatbot, max_tokens, temperature], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, max_tokens, temperature], [msg, chatbot])
        clear.click(lambda: ("", []), None, [msg, chatbot])

        gr.Markdown(
            f"""
            ---
            **Run locally:** `python app_ui_project1.py` · **Port:** {PORT} ·
            **Project 2 UI:** [localhost:7860](http://localhost:7860) ·
            **Repo:** [modern-llm-finetuning-lab](https://github.com/sophalchan/modern-llm-finetuning-lab)
            """
        )
    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)


if __name__ == "__main__":
    main()
