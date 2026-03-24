"""CLIエントリーポイント

使い方:
  python comfyui_cli.py list
  python comfyui_cli.py convert --input <workflow.json> --workflow-id <id> --name <name>
  python comfyui_cli.py generate --workflow-id <id> --prompt "..."
"""

from core.generate_image_core import main


if __name__ == "__main__":
    raise SystemExit(main())
