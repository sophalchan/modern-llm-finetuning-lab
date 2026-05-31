#!/usr/bin/env bash
# Launch Project 1 Gradio UI (QLoRA) on port 7861
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
exec python app_ui_project1.py "$@"
