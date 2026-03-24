"""ComfyUI CLI ベース画像生成・ワークフロー変換ユーティリティ。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

from core.comfyui_generator import (
    ComfyUIGenerator,
    PromptInjector,
    WorkflowLoader,
    WorkflowMeta,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKFLOW_ID = "txt2img_basic"
DEFAULT_NEGATIVE_PROMPT = (
    "ugly, deformed, noisy, blurry, low contrast, text, watermark, "
    "bad anatomy, worst quality, low quality, jpeg artifacts"
)


class _ProgressBar:
    def __init__(self) -> None:
        self.last_percent = -1.0
        self.active = False

    def _bar_width(self) -> int:
        cols = shutil.get_terminal_size((100, 20)).columns
        return max(20, min(50, cols - 25))

    def update(self, percent: float) -> None:
        p = max(0.0, min(100.0, float(percent)))
        if abs(p - self.last_percent) < 0.1:
            return
        self.active = True
        self.last_percent = p
        width = self._bar_width()
        filled = int(width * p / 100)
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r進捗 |{bar}| {p:5.1f}%")
        sys.stdout.flush()

    def newline(self) -> None:
        if self.active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.active = False

    def finish(self) -> None:
        if self.last_percent < 100:
            self.update(100.0)
        self.newline()


def _coerce_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_extra_params(items: list[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"extra_params の形式が不正です: {item} (key=value で指定してください)")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"extra_params の key が空です: {item}")
        params[key] = _coerce_value(value)
    return params


async def _run_generation(
    positive_prompt: str,
    negative_prompt: str,
    workflow_id: str,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generator = ComfyUIGenerator()
    progress = _ProgressBar()

    # ComfyUI が起動していない場合、少し待って再確認
    start_result = await generator.ensure_comfyui_running(start_timeout=15, poll_interval=3)
    if not start_result["success"]:
        return {"error": start_result["message"]}

    generated_files: list[str] = []
    error_message: Optional[str] = None

    async for event in generator.generate(
        workflow_id=workflow_id,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        extra_params=extra_params or {},
    ):
        etype = event.get("type", "")
        if etype == "status":
            progress.newline()
            print(f"[INFO] {event.get('message', '')}")
        elif etype == "progress":
            progress.update(event.get("percent", 0))
        elif etype == "file_generated":
            path = event.get("path", "")
            if path:
                generated_files.append(path)
                progress.newline()
                print(f"[INFO] 生成ファイル: {path}")
        elif etype == "error":
            error_message = event.get("message", "不明なエラー")
            progress.newline()
            print(f"[ERROR] {error_message}")
        elif etype == "finished":
            progress.finish()

    progress.newline()

    if error_message and not generated_files:
        return {"error": error_message}
    if not generated_files:
        return {"error": "画像が生成されませんでした。ワークフロー設定を確認してください。"}

    file_path = Path(generated_files[0])
    if not file_path.exists():
        return {"error": f"生成ファイルが見つかりません: {file_path}"}

    return {
        "type": "image",
        "data": file_path.read_bytes(),
        "filename": file_path.name,
        "path": str(file_path),
        "total_files": len(generated_files),
    }


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """ComfyUI で画像を生成して結果を返す。"""
    if not prompt.strip():
        return {"error": "prompt は必須です"}

    neg = negative_prompt.strip() if negative_prompt else DEFAULT_NEGATIVE_PROMPT
    result = asyncio.run(_run_generation(prompt.strip(), neg, workflow_id, extra_params))
    if result.get("error"):
        return {"error": result["error"]}
    return {
        "type": "image",
        "data": result["data"],
        "filename": result["filename"],
        "path": result["path"],
        "message": f"生成成功: {result['filename']}",
    }


async def _convert_workflow_async(
    input_path: str,
    workflow_id: str,
    name: str,
    description: str,
    output_type: str,
    output_format: str,
) -> Dict[str, Any]:
    loader = WorkflowLoader()
    src = Path(input_path)
    if not src.exists():
        return {"error": f"入力ファイルが見つかりません: {input_path}"}

    with src.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    workflow = raw_data
    if WorkflowLoader.is_ui_format(raw_data):
        # 可能なら object_info 付きで変換
        object_info = None
        try:
            gen = ComfyUIGenerator()
            if await gen.client.ping(timeout=3):
                object_info = await gen.client.get_object_info()
        except Exception:
            object_info = None
        workflow = WorkflowLoader.convert_ui_to_api(raw_data, object_info)

    prompt_nodes = PromptInjector.find_prompt_nodes(workflow)
    meta = WorkflowMeta(
        name=name,
        description=description,
        output_type=output_type,
        output_format=output_format,
        positive_node_id=prompt_nodes.get("positive"),
        negative_node_id=prompt_nodes.get("negative"),
        positive_field="text",
        negative_field="text",
        param_map={},
        tags=["converted"],
    )

    loader.save_workflow(workflow_id, workflow, meta)
    return {
        "success": True,
        "workflow_id": workflow_id,
        "output_dir": str((loader.workflows_dir / workflow_id).resolve()),
        "is_ui_source": WorkflowLoader.is_ui_format(raw_data),
    }


def convert_workflow(
    input_path: str,
    workflow_id: str,
    name: str,
    description: str = "",
    output_type: str = "image",
    output_format: str = "png",
) -> Dict[str, Any]:
    """ComfyUIからDLしたワークフロー(JSON)を生成用形式として保存する。"""
    return asyncio.run(
        _convert_workflow_async(
            input_path=input_path,
            workflow_id=workflow_id,
            name=name,
            description=description,
            output_type=output_type,
            output_format=output_format,
        )
    )


def list_workflows() -> list[Dict[str, Any]]:
    """保存済みワークフロー一覧を返す。"""
    return ComfyUIGenerator().list_workflows()


def interrupt_generation(client_id: Optional[str] = None) -> Dict[str, Any]:
    """実行中の生成を中断する。"""
    return asyncio.run(ComfyUIGenerator().interrupt(client_id=client_id))


def clear_generation_queue() -> Dict[str, Any]:
    """キューをクリアする。"""
    return asyncio.run(ComfyUIGenerator().clear_queue())


def get_generation_queue_info() -> Dict[str, Any]:
    """キュー情報を取得する。"""
    return asyncio.run(ComfyUIGenerator().get_queue_info())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ComfyUI CLI Workflow Builder")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="利用可能ワークフロー一覧")
    p_list.set_defaults(command="list")

    p_convert = sub.add_parser("convert", help="DLしたworkflow.jsonを変換して保存")
    p_convert.add_argument("--input", required=True, help="入力 JSON パス")
    p_convert.add_argument("--workflow-id", required=True, help="保存先ID")
    p_convert.add_argument("--name", default="converted workflow", help="ワークフロー名")
    p_convert.add_argument("--description", default="", help="説明")
    p_convert.add_argument("--output-type", default="image", choices=["image", "video"])
    p_convert.add_argument("--output-format", default="png")

    p_gen = sub.add_parser("generate", help="画像生成を実行")
    p_gen.add_argument("--prompt", required=True, help="ポジティブプロンプト")
    p_gen.add_argument("--negative", default="", help="ネガティブプロンプト")
    p_gen.add_argument("--workflow-id", default=DEFAULT_WORKFLOW_ID, help="使用ワークフローID")
    p_gen.add_argument(
        "--param",
        action="append",
        default=[],
        help="追加パラメータ key=value。複数指定可（例: --param seed=1 --param steps=20）",
    )

    p_interrupt = sub.add_parser("interrupt", help="実行中ジョブを中断")
    p_interrupt.add_argument(
        "--client-id",
        default=None,
        help="特定クライアントのみ中断する場合の client_id（省略時は全体）",
    )

    sub.add_parser("clear-queue", help="キューを全クリア")
    sub.add_parser("queue-info", help="キュー状態を表示")

    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        workflows = list_workflows()
        if not workflows:
            print("ワークフローがありません")
            return 0
        for wf in workflows:
            print(f"- {wf.get('id')} | {wf.get('name', '')} | type={wf.get('output_type', 'image')}")
        return 0

    if args.command == "convert":
        result = convert_workflow(
            input_path=args.input,
            workflow_id=args.workflow_id,
            name=args.name,
            description=args.description,
            output_type=args.output_type,
            output_format=args.output_format,
        )
        if result.get("error"):
            print(f"変換失敗: {result['error']}")
            return 1
        print(
            f"変換成功: workflow_id={result['workflow_id']} "
            f"保存先={result['output_dir']}"
        )
        return 0

    if args.command == "generate":
        try:
            extra_params = _parse_extra_params(args.param)
        except ValueError as exc:
            print(str(exc))
            return 1

        result = generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative,
            workflow_id=args.workflow_id,
            extra_params=extra_params,
        )
        if result.get("error"):
            print(f"生成失敗: {result['error']}")
            return 1

        print(f"生成成功: {result.get('path', '')}")
        return 0

    if args.command == "interrupt":
        result = interrupt_generation(client_id=args.client_id)
        if result.get("success"):
            print(f"中断成功: {result.get('message', '')}")
            return 0
        print(f"中断失敗: {result.get('message', '不明なエラー')}")
        return 1

    if args.command == "clear-queue":
        result = clear_generation_queue()
        if result.get("success"):
            print(f"キュークリア成功: {result.get('message', '')}")
            return 0
        print(f"キュークリア失敗: {result.get('message', '不明なエラー')}")
        return 1

    if args.command == "queue-info":
        result = get_generation_queue_info()
        if result.get("error"):
            print(f"キュー情報取得失敗: {result['error']}")
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
