#!/usr/bin/env python3
"""Gradio web UI for Project 2 ORPO/DoRA fine-tuned model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr

from src.inference_engine import get_engine
from src.utils import require_cuda

EXAMPLES = [
    "Give three bullet points for secure Active Directory operations.",
    "Explain incident response best practices in simple language.",
    "What is ORPO fine-tuning and why use it?",
    "How should a SOC team handle a phishing alert?",
]


def chat(message: str, history: list, max_tokens: int, temperature: float) -> str:
    if not message.strip():
        return "Please enter a question."
    try:
        engine = get_engine()
        return engine.generate(message, max_new_tokens=int(max_tokens), temperature=float(temperature))
    except Exception as exc:
        return f"Error: {exc}"


def build_ui() -> gr.Blocks:
    meta = {}
    try:
        require_cuda()
        meta = get_engine().info()
        status = f"Ready — {meta['base_model']} ({meta['method']})"
    except Exception as exc:
        status = f"Model not loaded: {exc}"

    with gr.Blocks(title="ORPO Fine-Tuned Assistant") as demo:
        gr.Markdown(
            """
            # Modern LLM Fine-Tuning Lab — Web UI
            **Project 2:** Qwen + **ORPO** + **DoRA** preference-aligned assistant (local GPU)

            Ask questions about cybersecurity, IT operations, AI/ML, or general topics.
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
            reply = chat(user_message, chat_history, mt, temp)
            chat_history = chat_history + [(user_message, reply)]
            return "", chat_history

        send.click(respond, [msg, chatbot, max_tokens, temperature], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, max_tokens, temperature], [msg, chatbot])
        clear.click(lambda: ("", []), None, [msg, chatbot])

        gr.Markdown(
            """
            ---
            **Run locally:** `python app_ui.py` · **Port:** 7860 · **Repo:** [modern-llm-finetuning-lab](https://github.com/sophalchan/modern-llm-finetuning-lab)
            """
        )
    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
