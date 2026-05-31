#!/usr/bin/env python3
"""Launch Gradio UI (wrapper)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "app_ui.py"), *sys.argv[1:]], check=True)
