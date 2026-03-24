"""
ComfyUIサーバーとの通信クライアント

WebSocket経由で生成進捗を監視し、
完成したファイルをローカルの設定フォルダ（COMFYUI_OUTPUT_DIR）に保存します。
"""
import json
import asyncio
import traceback
import requests
import websockets
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List

from config.settings import (
    COMFYUI_HOST,
    COMFYUI_PORT,
    COMFYUI_API_URL,
    COMFYUI_OUTPUT_DIR,
)

# --------------------------------------------------------------------------- #
# 出力ノードの class_type 定義
# --------------------------------------------------------------------------- #
IMAGE_OUTPUT_NODE_TYPES: frozenset = frozenset({
    "SaveImage",
    "SaveImageWebsocket",
    "Image Save",
    "WAS_Image_Save",
})

VIDEO_OUTPUT_NODE_TYPES: frozenset = frozenset({
    "VHS_VideoCombine",
    "SaveAnimatedWEBP",
    "SaveAnimatedPNG",
    "WAS_Video_Save",
})

# KSampler 系ノードの class_type（進捗監視用）
KSAMPLER_NODE_TYPES: frozenset = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "KSamplerSelect",
})


class ComfyUIClient:
    """
    ComfyUIサーバーとの通信を担うクライアント。

    - HTTP POST /prompt  : ワークフローをキューに投入
    - WebSocket /ws      : 生成進捗と完了イベントを受信
    - HTTP GET  /view    : 生成ファイルをダウンロード
    - ローカル保存        : COMFYUI_OUTPUT_DIR 以下に images/ または videos/ に格納
    """

    def __init__(self) -> None:
        self.host = COMFYUI_HOST
        self.port = COMFYUI_PORT
        self.api_url = COMFYUI_API_URL
        self.output_dir = Path(COMFYUI_OUTPUT_DIR)
        # 保存ディレクトリを事前に作成
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(parents=True, exist_ok=True)
        # 実行中の生成タスク: client_id -> prompt_id
        self.active_generations: Dict[str, str] = {}

    # ----------------------------------------------------------------------- #
    # 内部ユーティリティ
    # ----------------------------------------------------------------------- #

    def _detect_output_nodes(
        self, workflow: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """ワークフローから画像・動画の出力ノードIDを検出して返す"""
        image_nodes: List[str] = []
        video_nodes: List[str] = []
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get("class_type", "")
            if class_type in IMAGE_OUTPUT_NODE_TYPES:
                image_nodes.append(node_id)
            elif class_type in VIDEO_OUTPUT_NODE_TYPES:
                video_nodes.append(node_id)
        return {"image": image_nodes, "video": video_nodes}

    def _detect_ksampler_nodes(self, workflow: Dict[str, Any]) -> List[str]:
        """ワークフローからKSampler系ノードIDを検出して返す"""
        return [
            nid
            for nid, nd in workflow.items()
            if isinstance(nd, dict) and nd.get("class_type") in KSAMPLER_NODE_TYPES
        ]

    async def _download_file(
        self, filename: str, subfolder: str, file_type: str
    ) -> Optional[bytes]:
        """ComfyUIサーバーからファイルをダウンロードしてバイト列を返す"""
        from urllib.parse import quote

        # filename / subfolder をパーツごとに URL エンコード
        encoded_filename = quote(filename, safe="")
        encoded_subfolder = quote(subfolder, safe="/")
        url = (
            f"{self.api_url}/view"
            f"?filename={encoded_filename}"
            f"&subfolder={encoded_subfolder}"
            f"&type={file_type}"
        )
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, timeout=120)
            )
            response.raise_for_status()
            return response.content
        except Exception as exc:
            print(f"❌ ファイルダウンロードエラー ({filename}): {exc}")
            return None

    async def _fetch_history_outputs(
        self, prompt_id: str, save_node_ids: set
    ) -> List[Dict[str, Any]]:
        """
        /api/history/{prompt_id} から出力ファイル情報を取得する。
        WebSocket の executed イベントで %date:...% テンプレートが展開されない場合の
        フォールバック用。展開済みの実ファイル名が返る。

        Returns:
            list of {"filename": str, "subfolder": str, "type": str, "output_type": "image"|"video"}
        """
        url = f"{self.api_url}/history/{prompt_id}"
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, timeout=30)
            )
            response.raise_for_status()
            history = response.json()
        except Exception as exc:
            print(f"⚠️ history 取得失敗: {exc}")
            return []

        prompt_data = history.get(prompt_id, {})
        outputs = prompt_data.get("outputs", {})
        files: List[Dict[str, Any]] = []

        for node_id, node_output in outputs.items():
            if save_node_ids and node_id not in save_node_ids:
                continue
            for img in node_output.get("images", []):
                files.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                    "output_type": "image",
                })
            for key in ("gifs", "videos"):
                for vid in node_output.get(key, []):
                    files.append({
                        "filename": vid["filename"],
                        "subfolder": vid.get("subfolder", ""),
                        "type": vid.get("type", "output"),
                        "output_type": "video",
                    })
        return files

    def _save_file(
        self, content: bytes, output_type: str, original_filename: str
    ) -> str:
        """
        ファイルをローカルフォルダ（COMFYUI_OUTPUT_DIR）に保存してパスを返す。
        output_type: "image" | "video"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = Path(original_filename).suffix
        if not suffix:
            suffix = ".png" if output_type == "image" else ".mp4"
        save_dir = self.output_dir / ("images" if output_type == "image" else "videos")
        save_path = save_dir / f"{timestamp}{suffix}"
        save_path.write_bytes(content)
        print(f"保存完了: {save_path}")
        return str(save_path)

    # ----------------------------------------------------------------------- #
    # キュー投入
    # ----------------------------------------------------------------------- #

    async def queue_prompt(
        self, workflow: Dict[str, Any], client_id: str
    ) -> Optional[str]:
        """
        ワークフローをComfyUIのキューに投入し、prompt_id を返す。
        失敗時は None を返す。
        """
        payload = {"prompt": workflow, "client_id": client_id}
        headers = {"Content-Type": "application/json"}
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.api_url}/prompt",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60,
                ),
            )
            if response.status_code != 200:
                print(
                    f"❌ ComfyUIエラー (HTTP {response.status_code}): "
                    f"{response.text[:400]}"
                )
                return None
            data = response.json()
            # ComfyUI がバリデーションエラーを返す場合
            if "error" in data:
                print(f"❌ ワークフローエラー: {data['error']}")
                return None
            return data.get("prompt_id")
        except Exception as exc:
            print(f"❌ queue_prompt 例外: {exc}")
            return None

    # ----------------------------------------------------------------------- #
    # メイン: ストリーミング生成
    # ----------------------------------------------------------------------- #

    async def stream_generation(
        self,
        workflow: Dict[str, Any],
        client_id: str,
        output_type: str = "image",
        save_node_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        ComfyUIの生成プロセスをWebSocket経由でストリーミングし、
        イベントを dict で yield する。

        Parameters
        ----------
        workflow      : ComfyUI API形式のワークフロー辞書
        client_id     : クライアント識別子（uuid4推奨）
        output_type   : "image" | "video"
        save_node_ids : 監視する出力ノードIDリスト（None=自動検出）

        Yield されるイベント例
        ---------------------
        {"type": "status",        "message": "キューに追加されました", "prompt_id": "..."}
        {"type": "progress",      "node": "3", "current": 5, "total": 20, "percent": 25.0}
        {"type": "file_generated","output_type": "image", "path": "/...", "filename": "..."}
        {"type": "warning",       "message": "..."}
        {"type": "error",         "message": "..."}
        {"type": "finished",      "total_files": 1}
        """
        # --- 出力ノードを決定 ---
        if save_node_ids is None:
            detected = self._detect_output_nodes(workflow)
            save_node_ids = detected.get(output_type, [])
            if not save_node_ids:
                # フォールバック: 画像・動画すべての出力ノードを監視
                save_node_ids = detected["image"] + detected["video"]

        if not save_node_ids:
            yield {
                "type": "error",
                "message": (
                    "ワークフロー内に出力ノードが見つかりませんでした。"
                    "SaveImage / VHS_VideoCombine 等のノードを確認してください。"
                ),
            }
            return

        ksampler_ids = set(self._detect_ksampler_nodes(workflow))
        save_node_ids_set = set(save_node_ids)

        ws_url = f"ws://{self.host}:{self.port}/ws?clientId={client_id}"
        generated_count = 0
        prompt_id: Optional[str] = None
        no_event_timeouts = 0
        generation_started = False
        last_queue_remaining: Optional[int] = None

        try:
            async with websockets.connect(
                ws_url, ping_interval=30, ping_timeout=10
            ) as ws:
                # --- キュー投入 ---
                # WS接続後に投入することで、開始直後イベントの取りこぼしを防ぐ
                prompt_id = await self.queue_prompt(workflow, client_id)
                if not prompt_id:
                    yield {"type": "error", "message": "ComfyUIへのジョブ投入に失敗しました"}
                    return

                self.active_generations[client_id] = prompt_id
                yield {
                    "type": "status",
                    "message": "キューに追加されました",
                    "prompt_id": prompt_id,
                    "client_id": client_id,
                }

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        msg = json.loads(raw)
                        no_event_timeouts = 0
                    except asyncio.TimeoutError:
                        no_event_timeouts += 1
                        yield {
                            "type": "warning",
                            "message": f"WebSocket受信タイムアウト（{no_event_timeouts}回目）",
                        }

                        # タイムアウト時は history を確認して完了済みファイルがあれば回収して終了
                        if prompt_id:
                            history_files = await self._fetch_history_outputs(prompt_id, save_node_ids_set)
                            if history_files:
                                for finfo in history_files:
                                    file_bytes = await self._download_file(
                                        finfo["filename"], finfo["subfolder"], finfo["type"]
                                    )
                                    if file_bytes:
                                        save_path = self._save_file(
                                            file_bytes, finfo["output_type"], finfo["filename"]
                                        )
                                        generated_count += 1
                                        yield {
                                            "type": "file_generated",
                                            "output_type": finfo["output_type"],
                                            "path": save_path,
                                            "filename": Path(save_path).name,
                                            "index": generated_count,
                                        }
                                break

                        # 連続タイムアウトが続く場合は終了
                        if no_event_timeouts >= 3:
                            yield {
                                "type": "error",
                                "message": "ComfyUIイベントを受信できません。処理を中断しました。",
                            }
                            break
                        continue
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")
                    data = msg.get("data", {})
                    msg_prompt_id = data.get("prompt_id")

                    # ----- status -----
                    if msg_type == "status":
                        # status は高頻度で飛ぶため、待機中のみ・値変化時のみ通知
                        if generation_started:
                            continue

                        q = (
                            data.get("status", {})
                            .get("exec_info", {})
                            .get("queue_remaining", 0)
                        )

                        if q == last_queue_remaining:
                            continue
                        last_queue_remaining = q

                        yield {"type": "status", "message": f"キュー残り: {q}件"}

                    # ----- execution_start -----
                    elif msg_type == "execution_start":
                        if msg_prompt_id == prompt_id:
                            generation_started = True
                            yield {"type": "status", "message": "生成を開始します"}

                    # ----- progress -----
                    elif msg_type == "progress" and msg_prompt_id == prompt_id:
                        node_id = data.get("node")
                        if not ksampler_ids or node_id in ksampler_ids:
                            cur = data.get("value", 0)
                            total = data.get("max", 1)
                            pct = round(cur / total * 100, 1) if total > 0 else 0.0
                            yield {
                                "type": "progress",
                                "node": node_id,
                                "current": cur,
                                "total": total,
                                "percent": pct,
                            }

                    # ----- executed (出力ファイルを受信) -----
                    elif msg_type == "executed" and msg_prompt_id == prompt_id:
                        node_id = data.get("node")
                        outputs = data.get("output", {})

                        if node_id not in save_node_ids_set:
                            continue

                        # 画像ファイル
                        for image_data in outputs.get("images", []):
                            fname = image_data["filename"]
                            subfolder = image_data.get("subfolder", "")
                            ftype = image_data.get("type", "output")

                            file_bytes = await self._download_file(fname, subfolder, ftype)
                            if file_bytes:
                                save_path = self._save_file(file_bytes, "image", fname)
                                generated_count += 1
                                yield {
                                    "type": "file_generated",
                                    "output_type": "image",
                                    "path": save_path,
                                    "filename": Path(save_path).name,
                                    "index": generated_count,
                                }
                            else:
                                print(f"WebSocket経由のダウンロード失敗 ({fname})、history フォールバックで再試行予定")

                        # 動画ファイル（VHS_VideoCombine は "gifs" キーに格納）
                        for key in ("gifs", "videos", "images"):
                            if key == "images":
                                continue  # 上で処理済み
                            for video_data in outputs.get(key, []):
                                fname = video_data["filename"]
                                subfolder = video_data.get("subfolder", "")
                                ftype = video_data.get("type", "output")

                                file_bytes = await self._download_file(fname, subfolder, ftype)
                                if file_bytes:
                                    save_path = self._save_file(file_bytes, "video", fname)
                                    generated_count += 1
                                    yield {
                                        "type": "file_generated",
                                        "output_type": "video",
                                        "path": save_path,
                                        "filename": Path(save_path).name,
                                        "index": generated_count,
                                    }
                                else:
                                    print(f"WebSocket経由のダウンロード失敗 ({fname})、history フォールバックで再試行予定")

                    # ----- executing node=None → 完了 -----
                    elif msg_type == "executing" and msg_prompt_id == prompt_id:
                        if data.get("node") is None:
                            yield {"type": "status", "message": "全ノードの実行が完了しました"}
                            break

                    # ----- execution_error -----
                    elif msg_type == "execution_error" and msg_prompt_id == prompt_id:
                        err_msg = data.get("exception_message", "不明なエラー")
                        yield {"type": "error", "message": f"実行エラー: {err_msg}"}
                        break

            # --- フォールバック: WebSocket 経由でファイル取得できなかった場合 ---
            if generated_count == 0 and prompt_id:
                print("WebSocket経由のダウンロードが0件のため /api/history からファイル取得を試みます...")
                await asyncio.sleep(1)  # history が反映されるまで少し待つ
                history_files = await self._fetch_history_outputs(prompt_id, save_node_ids_set)
                for finfo in history_files:
                    file_bytes = await self._download_file(
                        finfo["filename"], finfo["subfolder"], finfo["type"]
                    )
                    if file_bytes:
                        save_path = self._save_file(file_bytes, finfo["output_type"], finfo["filename"])
                        generated_count += 1
                        yield {
                            "type": "file_generated",
                            "output_type": finfo["output_type"],
                            "path": save_path,
                            "filename": Path(save_path).name,
                            "index": generated_count,
                        }
                    else:
                        yield {
                            "type": "error",
                            "message": f"history フォールバックでもダウンロード失敗: {finfo['filename']}",
                        }

        except websockets.exceptions.ConnectionClosed as exc:
            yield {"type": "error", "message": f"WebSocket接続が閉じられました: {exc}"}
        except Exception as exc:
            traceback.print_exc()
            yield {"type": "error", "message": f"予期しないエラー: {exc}"}
        finally:
            self.active_generations.pop(client_id, None)
            yield {"type": "finished", "total_files": generated_count}

    # ----------------------------------------------------------------------- #
    # キュー操作
    # ----------------------------------------------------------------------- #

    async def interrupt_generation(
        self, client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """現在実行中の生成を中断する"""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.api_url}/interrupt", timeout=10),
            )
            if client_id:
                self.active_generations.pop(client_id, None)
            else:
                self.active_generations.clear()
            return {"success": True, "message": "生成を停止しました"}
        except Exception as exc:
            return {"success": False, "message": f"停止エラー: {exc}"}

    async def clear_queue(self) -> Dict[str, Any]:
        """ComfyUIのキューを全クリアする"""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.api_url}/queue",
                    json={"clear": True},
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                ),
            )
            self.active_generations.clear()
            return {"success": True, "message": "キューをクリアしました"}
        except Exception as exc:
            return {"success": False, "message": f"クリアエラー: {exc}"}

    async def get_queue_info(self) -> Dict[str, Any]:
        """キューの現在状態を取得する"""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{self.api_url}/queue", timeout=10),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return {"error": str(exc)}

    async def ping(self, timeout: int = 5) -> bool:
        """
        ComfyUI HTTP API への疎通確認。
        /system_stats に GET して 200 が返れば True。
        """
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{self.api_url}/system_stats", timeout=timeout),
            )
            return response.status_code == 200
        except Exception:
            return False

    # ----------------------------------------------------------------------- #
    # 画像アップロード (img2img / ControlNet 用)
    # ----------------------------------------------------------------------- #

    async def upload_image(
        self, image_data: bytes, filename: str
    ) -> Optional[str]:
        """
        画像をComfyUIサーバーにアップロードし、
        サーバー上のファイル名を返す（img2img / ControlNet 等で使用）。
        """
        loop = asyncio.get_event_loop()
        try:
            files = {"image": (filename, BytesIO(image_data), "image/png")}
            data = {"type": "input", "subfolder": ""}
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.api_url}/upload/image",
                    files=files,
                    data=data,
                    timeout=60,
                ),
            )
            response.raise_for_status()
            return response.json().get("name")
        except Exception as exc:
            print(f"画像アップロードエラー: {exc}")
            return None

    # ----------------------------------------------------------------------- #
    # サーバー情報取得
    # ----------------------------------------------------------------------- #

    async def get_object_info(self) -> Dict[str, Any]:
        """
        ComfyUIサーバーから全ノードタイプの定義情報を取得する。
        UI形式→API形式変換時のウィジェットマッピングに使用。
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{self.api_url}/object_info", timeout=30),
        )
        response.raise_for_status()
        return response.json()

    async def get_models(self) -> Dict[str, Any]:
        """
        ComfyUIから利用可能なモデル（チェックポイント・LoRA・サンプラー等）を取得する。
        """
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{self.api_url}/object_info", timeout=30),
            )
            response.raise_for_status()
            data = response.json()

            models: Dict[str, List[str]] = {
                "checkpoints": [],
                "loras": [],
                "samplers": [],
                "schedulers": [],
                "upscalers": [],
            }

            def _extract_list(node_name: str, field: str) -> List[str]:
                info = data.get(node_name, {})
                raw = info.get("input", {}).get("required", {}).get(field, [[]])
                if isinstance(raw, list) and raw:
                    inner = raw[0]
                    return inner if isinstance(inner, list) else [x for x in raw if isinstance(x, str)]
                return []

            models["checkpoints"] = _extract_list("CheckpointLoaderSimple", "ckpt_name")
            models["loras"] = _extract_list("LoraLoader", "lora_name")
            models["samplers"] = _extract_list("KSampler", "sampler_name")
            models["schedulers"] = _extract_list("KSampler", "scheduler")
            models["upscalers"] = _extract_list("UpscaleModelLoader", "model_name")

            print(
                f"モデル取得: チェックポイント={len(models['checkpoints'])}件, "
                f"LoRA={len(models['loras'])}件, "
                f"サンプラー={len(models['samplers'])}件"
            )
            return models
        except Exception as exc:
            print(f"モデル取得エラー: {exc}")
            return {"checkpoints": [], "loras": [], "samplers": [], "schedulers": [], "upscalers": []}
