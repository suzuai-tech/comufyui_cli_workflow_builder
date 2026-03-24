"""プロジェクト設定。環境変数で上書き可能。"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "localhost")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", f"http://{COMFYUI_HOST}:{COMFYUI_PORT}")

COMFYUI_WORKFLOWS_DIR = os.getenv(
    "COMFYUI_WORKFLOWS_DIR", str(BASE_DIR / "comfyui_workflows")
)
COMFYUI_OUTPUT_DIR = os.getenv("COMFYUI_OUTPUT_DIR", str(BASE_DIR / "images"))
