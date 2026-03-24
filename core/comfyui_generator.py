"""
ComfyUI ワークフロー管理・プロンプト注入・生成制御

構成
----
WorkflowMeta    : ワークフローのメタデータ（出力タイプ、プロンプトノード指定、etc.）
WorkflowLoader  : ワークフロー JSON の読み込み・保存
PromptInjector  : ワークフローへのプロンプト・パラメータ注入（プロンプトノード自動検出付き）
ComfyUIGenerator: 高レベル生成インターフェース

ワークフローのディレクトリ構造
------------------------------
data/comfyui_workflows/
    <workflow_id>/
        workflow.json   # ComfyUI API 形式のワークフロー
        meta.json       # 任意のメタデータ（後述）

meta.json の例
--------------
{
    "name": "基本テキスト→画像",
    "description": "SD1.5 系の基本 txt2img ワークフロー",
    "output_type": "image",          // "image" | "video"
    "output_format": "png",          // "png" | "jpg" | "webp" | "mp4" | "gif"
    "positive_node_id": null,        // null = 自動検出
    "negative_node_id": null,
    "positive_field": "text",
    "negative_field": "text",
    "param_map": {
        "seed":         {"node_id": "3", "field": "seed"},
        "steps":        {"node_id": "3", "field": "steps"},
        "cfg":          {"node_id": "3", "field": "cfg"},
        "sampler_name": {"node_id": "3", "field": "sampler_name"},
        "scheduler":    {"node_id": "3", "field": "scheduler"},
        "width":        {"node_id": "5", "field": "width"},
        "height":       {"node_id": "5", "field": "height"},
        "checkpoint":   {"node_id": "4", "field": "ckpt_name"}
    },
    "tags": ["text2image", "basic"]
}

extra_params の指定方法
-----------------------
generate() / generate_from_file() の extra_params 引数で渡す辞書。

  1. meta.json の param_map キー名で指定:
       {"seed": 42, "steps": 20, "cfg": 7.5}

  2. "node_id:field" 形式で直接指定:
       {"3:seed": 42, "5:width": 768}

  3. KSampler ノードが持つフィールド名で指定（自動検索）:
       {"seed": 12345}  → KSampler の seed フィールドを書き換え
"""
import json
import copy
import uuid
import asyncio
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, AsyncGenerator

from config.settings import COMFYUI_WORKFLOWS_DIR

# --------------------------------------------------------------------------- #
# ノード class_type 定数
# --------------------------------------------------------------------------- #
KSAMPLER_CLASS_TYPES: frozenset = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "KSamplerSelect",
    "ADE_AnimateDiffSamplerSettings",
})

CLIP_TEXT_ENCODE_TYPES: frozenset = frozenset({
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeSDXLRefiner",
})


# --------------------------------------------------------------------------- #
# WorkflowMeta
# --------------------------------------------------------------------------- #
@dataclass
class WorkflowMeta:
    """
    ワークフローのメタデータ。
    meta.json が存在しない場合はデフォルト値を使用し、
    プロンプトノードは PromptInjector が自動検出する。
    """
    name: str = ""
    description: str = ""
    output_type: str = "image"      # "image" | "video"
    output_format: str = "png"      # "png" | "jpg" | "webp" | "mp4" | "gif"
    # プロンプトノードの手動指定（None = 自動検出）
    positive_node_id: Optional[str] = None
    negative_node_id: Optional[str] = None
    positive_field: str = "text"
    negative_field: str = "text"
    # 追加パラメータのノードマッピング: {"param_name": {"node_id": "3", "field": "seed"}}
    param_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowMeta":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            output_type=data.get("output_type", "image"),
            output_format=data.get("output_format", "png"),
            positive_node_id=data.get("positive_node_id"),
            negative_node_id=data.get("negative_node_id"),
            positive_field=data.get("positive_field", "text"),
            negative_field=data.get("negative_field", "text"),
            param_map=data.get("param_map", {}),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "output_type": self.output_type,
            "output_format": self.output_format,
            "positive_node_id": self.positive_node_id,
            "negative_node_id": self.negative_node_id,
            "positive_field": self.positive_field,
            "negative_field": self.negative_field,
            "param_map": self.param_map,
            "tags": self.tags,
        }


# --------------------------------------------------------------------------- #
# WorkflowLoader
# --------------------------------------------------------------------------- #
class WorkflowLoader:
    """
    ワークフロー JSON とメタデータを読み書きするクラス。

    データ配置:
        <workflows_dir>/
            <workflow_id>/
                workflow.json
                meta.json      (任意)
    """

    def __init__(self, workflows_dir: Optional[str] = None) -> None:
        self.workflows_dir = Path(workflows_dir or COMFYUI_WORKFLOWS_DIR)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        利用可能なワークフロー一覧を返す。
        戻り値: [{"id": str, ...WorkflowMeta フィールド}, ...]
        """
        workflows = []
        for workflow_dir in sorted(self.workflows_dir.iterdir()):
            if not workflow_dir.is_dir():
                continue
            workflow_json = workflow_dir / "workflow.json"
            if not workflow_json.exists():
                continue
            meta = self._load_meta(workflow_dir, default_name=workflow_dir.name)
            workflows.append({"id": workflow_dir.name, **meta.to_dict()})
        return workflows

    def load_workflow(self, workflow_id: str) -> Tuple[Dict[str, Any], WorkflowMeta]:
        """
        workflow_id（フォルダ名）からワークフローとメタデータを読み込む。
        戻り値: (workflow_dict, WorkflowMeta)
        """
        workflow_dir = self.workflows_dir / workflow_id
        if not workflow_dir.is_dir():
            raise FileNotFoundError(
                f"ワークフローフォルダが見つかりません: {workflow_dir}"
            )
        workflow_json = workflow_dir / "workflow.json"
        if not workflow_json.exists():
            raise FileNotFoundError(
                f"workflow.json が見つかりません: {workflow_json}"
            )
        workflow = self._read_json(workflow_json)
        meta = self._load_meta(workflow_dir, default_name=workflow_id)
        return workflow, meta

    def load_from_file(self, json_path: str) -> Tuple[Dict[str, Any], WorkflowMeta]:
        """
        任意の場所にある JSON ファイルを直接読み込む。
        同じディレクトリに meta.json があれば読み込む。
        戻り値: (workflow_dict, WorkflowMeta)
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {json_path}")
        workflow = self._read_json(path)
        meta = self._load_meta(path.parent, default_name=path.stem)
        return workflow, meta

    def save_workflow(
        self,
        workflow_id: str,
        workflow: Dict[str, Any],
        meta: Optional[WorkflowMeta] = None,
    ) -> None:
        """ワークフローとメタデータを保存する"""
        workflow_dir = self.workflows_dir / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        with (workflow_dir / "workflow.json").open("w", encoding="utf-8") as f:
            json.dump(workflow, f, ensure_ascii=False, indent=2)
        if meta:
            with (workflow_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"ワークフロー保存: {workflow_dir}")

    # --- 内部ユーティリティ ---

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _load_meta(self, directory: Path, default_name: str = "") -> WorkflowMeta:
        meta_path = directory / "meta.json"
        if meta_path.exists():
            return WorkflowMeta.from_dict(self._read_json(meta_path))
        return WorkflowMeta(name=default_name)

    # ------------------------------------------------------------------- #
    # UI形式 → API形式 変換
    # ------------------------------------------------------------------- #

    @staticmethod
    def is_ui_format(data: Dict[str, Any]) -> bool:
        """データが ComfyUI の UI（litegraph）形式であるかを判定する"""
        return (
            isinstance(data.get("nodes"), list)
            and isinstance(data.get("links"), list)
        )

    @staticmethod
    def convert_ui_to_api(
        ui_data: Dict[str, Any],
        object_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ComfyUI の UI (litegraph) 形式を API 形式に変換する。

        object_info が提供されると、サーバー定義に基づいてウィジェット名を
        正確にマッピングする。提供されない場合は inputs の widget プロパティ
        を使用するフォールバックヒューリスティックを使う。
        """
        nodes = ui_data.get("nodes", [])
        links_list = ui_data.get("links", [])

        # リンクマップ: link_id -> [source_node_id_str, source_output_slot]
        link_map: Dict[int, list] = {}
        for link in links_list:
            link_map[link[0]] = [str(link[1]), link[2]]

        # 表示専用ノード（変換対象外）
        SKIP_TYPES = frozenset({
            "Note", "MarkdownNote", "Reroute", "PrimitiveNode",
        })

        # テンソル系タイプ（ウィジェットにならない）
        TENSOR_TYPES = frozenset({
            "MODEL", "CONDITIONING", "LATENT", "VAE", "CLIP",
            "CLIP_VISION", "CLIP_VISION_OUTPUT", "IMAGE", "MASK",
            "CONTROL_NET", "UPSCALE_MODEL", "SIGMAS", "NOISE",
            "GUIDER", "SAMPLER", "STYLE_MODEL", "GLIGEN",
            "PHOTOMAKER", "TAESD",
        })

        # seed 後に現れる隠しウィジェットの値
        SEED_CONTROL = frozenset({
            "fixed", "increment", "decrement", "randomize",
        })

        api_workflow: Dict[str, Any] = {}

        for node in nodes:
            node_id = str(node["id"])
            class_type = node.get("type", "")

            if class_type in SKIP_TYPES or not class_type:
                continue

            node_inputs = node.get("inputs", [])
            widgets_values = list(node.get("widgets_values") or [])

            api_inputs: Dict[str, Any] = {}
            connected_names: set = set()

            # ---- Step 1: コネクション（リンク）を処理 ----
            for inp in node_inputs:
                name = inp.get("name", "")
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    api_inputs[name] = list(link_map[link_id])
                    connected_names.add(name)

            # ---- Step 2: ウィジェット入力名の順序リストを構築 ----
            widget_names: List[str] = []

            if object_info and class_type in object_info:
                # サーバー定義を使って正確にマッピング
                node_def = object_info[class_type].get("input", {})
                for section in ("required", "optional"):
                    for input_name, spec in node_def.get(section, {}).items():
                        if not spec:
                            continue
                        type_info = spec[0]
                        # COMBO（リスト定義）またはスカラー型はウィジェット
                        if isinstance(type_info, list):
                            widget_names.append(input_name)
                        elif (
                            isinstance(type_info, str)
                            and type_info.upper() not in TENSOR_TYPES
                        ):
                            widget_names.append(input_name)
            else:
                # フォールバック: inputs の widget プロパティを使用
                for inp in node_inputs:
                    if "widget" in inp:
                        widget_names.append(inp["widget"]["name"])

            # ---- Step 3: widgets_values をウィジェット名にマッピング ----
            wi = 0  # widget_names index
            wv = 0  # widgets_values index

            while wv < len(widgets_values) and wi < len(widget_names):
                val = widgets_values[wv]

                # dict / list はカスタムウィジェット（LoRA設定等）→ スキップ
                if isinstance(val, (dict, list)):
                    wv += 1
                    continue

                name = widget_names[wi]
                if name not in connected_names:
                    api_inputs[name] = val

                wi += 1
                wv += 1

                # seed / noise_seed の後に隠し control_after_generate をスキップ
                if (
                    name.lower() in ("seed", "noise_seed")
                    and wv < len(widgets_values)
                ):
                    nval = widgets_values[wv]
                    if isinstance(nval, str) and nval in SEED_CONTROL:
                        wv += 1

            # ---- ノードを登録 ----
            entry: Dict[str, Any] = {
                "class_type": class_type,
                "inputs": api_inputs,
            }
            # 表示用タイトルがあれば _meta に保存
            title = node.get("title")
            if title and title != class_type:
                entry["_meta"] = {"title": title}

            api_workflow[node_id] = entry

        print(
            f"UI→API 変換完了: {len(api_workflow)} ノード "
            f"(元 {len(nodes)} ノード、"
            f"スキップ {len(nodes) - len(api_workflow)} ノード)"
        )
        return api_workflow


# --------------------------------------------------------------------------- #
# PromptInjector
# --------------------------------------------------------------------------- #
class PromptInjector:
    """
    ワークフロー辞書へのプロンプト・パラメータ注入。

    プロンプトノードの検出ロジック:
        1. KSampler 系ノードを探す
        2. inputs["positive"] / inputs["negative"] の参照先ノードIDを取得
        3. 参照先が CLIPTextEncode 系なら positive / negative ノードとして確定
    """

    @staticmethod
    def find_prompt_nodes(
        workflow: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        """
        ワークフローから positive / negative のプロンプトノードIDを自動検出する。
        戻り値: {"positive": node_id | None, "negative": node_id | None}
        """
        result: Dict[str, Optional[str]] = {"positive": None, "negative": None}

        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            if node_data.get("class_type") not in KSAMPLER_CLASS_TYPES:
                continue

            inputs = node_data.get("inputs", {})

            # positive 参照先を解決
            pos_ref = inputs.get("positive")
            if isinstance(pos_ref, list) and pos_ref:
                ref_id = str(pos_ref[0])
                if (
                    ref_id in workflow
                    and workflow[ref_id].get("class_type") in CLIP_TEXT_ENCODE_TYPES
                ):
                    result["positive"] = ref_id

            # negative 参照先を解決
            neg_ref = inputs.get("negative")
            if isinstance(neg_ref, list) and neg_ref:
                ref_id = str(neg_ref[0])
                if (
                    ref_id in workflow
                    and workflow[ref_id].get("class_type") in CLIP_TEXT_ENCODE_TYPES
                ):
                    result["negative"] = ref_id

            # 最初の KSampler で見つかったら終了
            if result["positive"] or result["negative"]:
                break

        return result

    @staticmethod
    def inject(
        workflow: Dict[str, Any],
        positive_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        meta: Optional[WorkflowMeta] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        ワークフローのディープコピーにプロンプトとパラメータを注入して返す。

        extra_params 指定方法
        ---------------------
        1. meta.param_map のキー名: {"seed": 42, "steps": 20}
        2. "node_id:field" 直接指定:  {"3:seed": 42}
        3. KSampler のフィールド名で自動マッチ: {"seed": 42}
           (meta.param_map になく ":" もない場合のフォールバック)
        """
        wf = copy.deepcopy(workflow)

        # --- プロンプトノードを決定 ---
        pos_node_id = meta.positive_node_id if meta else None
        neg_node_id = meta.negative_node_id if meta else None
        pos_field = meta.positive_field if meta else "text"
        neg_field = meta.negative_field if meta else "text"

        if pos_node_id is None or neg_node_id is None:
            detected = PromptInjector.find_prompt_nodes(wf)
            if pos_node_id is None:
                pos_node_id = detected["positive"]
            if neg_node_id is None:
                neg_node_id = detected["negative"]

        # --- プロンプトを注入 ---
        if positive_prompt is not None and pos_node_id and pos_node_id in wf:
            wf[pos_node_id]["inputs"][pos_field] = positive_prompt
            print(f"ポジティブプロンプト注入 → ノード {pos_node_id}[{pos_field}]")

        if negative_prompt is not None and neg_node_id and neg_node_id in wf:
            wf[neg_node_id]["inputs"][neg_field] = negative_prompt
            print(f"ネガティブプロンプト注入 → ノード {neg_node_id}[{neg_field}]")

        # --- 追加パラメータを注入 ---
        if extra_params:
            param_map = meta.param_map if meta else {}

            for key, value in extra_params.items():
                # 形式 1: "node_id:field" 直接指定
                if ":" in key:
                    node_id, field_name = key.split(":", 1)
                    if node_id in wf:
                        wf[node_id]["inputs"][field_name] = value
                        print(f"✅ 直接注入 [{key}] = {value}")
                    continue

                # 形式 2: param_map 経由
                if key in param_map:
                    mapping = param_map[key]
                    node_id = mapping.get("node_id", "")
                    field_name = mapping.get("field", "")
                    if node_id and field_name and node_id in wf:
                        wf[node_id]["inputs"][field_name] = value
                        print(f"param_map 注入 [{key}] → ノード {node_id}.{field_name} = {value}")
                    continue

                # 形式 3: KSampler フィールド自動検索
                for node_id, node_data in wf.items():
                    if not isinstance(node_data, dict):
                        continue
                    if node_data.get("class_type") in KSAMPLER_CLASS_TYPES:
                        if key in node_data.get("inputs", {}):
                            wf[node_id]["inputs"][key] = value
                            print(f"KSampler 自動注入 [{key}] → ノード {node_id} = {value}")
                            break

        return wf


# --------------------------------------------------------------------------- #
# ComfyUIGenerator
# --------------------------------------------------------------------------- #
class ComfyUIGenerator:
    """
    ComfyUI 生成の高レベルインターフェース。

    api.ComfyUIClient を内部で使用し、ワークフローの読み込みから
    プロンプト注入・生成実行・ファイル保存まで一括して管理する。

    使用例
    ------
    generator = ComfyUIGenerator()

    # ワークフロー ID で実行
    async for event in generator.generate(
        workflow_id="txt2img_basic",
        positive_prompt="1girl, masterpiece",
        negative_prompt="bad quality",
        extra_params={"seed": 42, "steps": 25},
    ):
        print(event)

    # 任意の JSON ファイルで実行
    async for event in generator.generate_from_file(
        json_path="/path/to/my_workflow.json",
        positive_prompt="landscape",
    ):
        print(event)
    """

    def __init__(self) -> None:
        from api.comfyui_client import ComfyUIClient

        self.client = ComfyUIClient()
        self.loader = WorkflowLoader()

    # ----------------------------------------------------------------------- #
    # ComfyUI 起動保証
    # ----------------------------------------------------------------------- #

    async def ensure_comfyui_running(
        self,
        start_timeout: int = 120,
        poll_interval: int = 5,
    ) -> Dict[str, Any]:
        """
        ComfyUI API が応答可能になるまで待機する。

        注意:
        - 本ツールでは systemctl/SSH による起動は行わない
        - 既に起動している ComfyUI への接続確認に特化
        """
        if await self.client.ping():
            return {
                "success": True,
                "message": "ComfyUI は起動中です",
                "was_started": False,
            }

        elapsed = 0
        while elapsed < start_timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if await self.client.ping():
                return {
                    "success": True,
                    "message": f"ComfyUI に接続できました ({elapsed}秒)",
                    "was_started": False,
                }

        return {
            "success": False,
            "message": (
                "ComfyUI API に接続できませんでした。"
                "先に ComfyUI を起動し、host/port 設定を確認してください。"
            ),
            "was_started": False,
        }

    # ----------------------------------------------------------------------- #
    # 公開 API
    # ----------------------------------------------------------------------- #

    async def generate(
        self,
        workflow_id: str,
        positive_prompt: str = "",
        negative_prompt: str = "",
        extra_params: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        ワークフロー ID を指定して生成を実行する。

        Parameters
        ----------
        workflow_id     : data/comfyui_workflows/ 以下のフォルダ名
        positive_prompt : ポジティブプロンプト（空文字でワークフロー既存値を維持）
        negative_prompt : ネガティブプロンプト
        extra_params    : 追加パラメータ（seed, steps, cfg, "node_id:field" 等）
        client_id       : WebSocket 識別子（省略時は自動生成）
        """
        workflow, meta = self.loader.load_workflow(workflow_id)
        async for event in self._run_generation(
            workflow, meta, positive_prompt, negative_prompt, extra_params, client_id
        ):
            yield event

    async def generate_from_file(
        self,
        json_path: str,
        positive_prompt: str = "",
        negative_prompt: str = "",
        output_type: str = "image",
        extra_params: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        任意の JSON ファイルパスを直接指定して生成を実行する。

        Parameters
        ----------
        json_path   : workflow.json の絶対パスまたは相対パス
        output_type : "image" | "video"
        """
        workflow, meta = self.loader.load_from_file(json_path)
        # output_type の明示指定を優先
        if output_type and output_type != "image":
            meta.output_type = output_type
        async for event in self._run_generation(
            workflow, meta, positive_prompt, negative_prompt, extra_params, client_id
        ):
            yield event

    async def generate_from_workflow(
        self,
        workflow: Dict[str, Any],
        positive_prompt: str = "",
        negative_prompt: str = "",
        output_type: str = "image",
        extra_params: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        ワークフロー辞書を直接渡して生成を実行する（動的生成用）。
        """
        meta = WorkflowMeta(output_type=output_type)
        async for event in self._run_generation(
            workflow, meta, positive_prompt, negative_prompt, extra_params, client_id
        ):
            yield event

    # ----------------------------------------------------------------------- #
    # キュー操作
    # ----------------------------------------------------------------------- #

    async def interrupt(
        self, client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """実行中の生成を中断する"""
        return await self.client.interrupt_generation(client_id)

    async def clear_queue(self) -> Dict[str, Any]:
        """ComfyUI のキューを全クリアする"""
        return await self.client.clear_queue()

    async def get_queue_info(self) -> Dict[str, Any]:
        """キューの現在状態を取得する"""
        return await self.client.get_queue_info()

    async def get_available_models(self) -> Dict[str, Any]:
        """利用可能なモデル情報を取得する"""
        return await self.client.get_models()

    # ----------------------------------------------------------------------- #
    # ワークフロー管理
    # ----------------------------------------------------------------------- #

    def list_workflows(self) -> List[Dict[str, Any]]:
        """利用可能なワークフロー一覧を返す"""
        return self.loader.list_workflows()

    def save_workflow(
        self,
        workflow_id: str,
        workflow: Dict[str, Any],
        meta: Optional[WorkflowMeta] = None,
    ) -> None:
        """ワークフローを保存する"""
        self.loader.save_workflow(workflow_id, workflow, meta)

    def get_prompt_node_ids(
        self, workflow: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        """
        ワークフローのポジティブ/ネガティブプロンプトノードIDを返す。
        デバッグや UI 表示用。
        """
        return PromptInjector.find_prompt_nodes(workflow)

    # ----------------------------------------------------------------------- #
    # 内部実装
    # ----------------------------------------------------------------------- #

    async def _run_generation(
        self,
        workflow: Dict[str, Any],
        meta: WorkflowMeta,
        positive_prompt: str,
        negative_prompt: str,
        extra_params: Optional[Dict[str, Any]],
        client_id: Optional[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """UI形式自動変換 → プロンプト注入 → ComfyUIClient へ委譲"""
        cid = client_id or str(uuid.uuid4())

        # UI形式の場合は API形式に変換
        if WorkflowLoader.is_ui_format(workflow):
            print("UI形式のワークフローを検出。API形式に変換します...")
            try:
                object_info = await self.client.get_object_info()
                workflow = WorkflowLoader.convert_ui_to_api(workflow, object_info)
            except Exception as exc:
                yield {
                    "type": "error",
                    "message": (
                        f"ワークフロー変換エラー: {exc}\n"
                        "ComfyUI の 'Save (API Format)' で書き出した JSON を"
                        "使用すると確実です"
                    ),
                }
                return

        # プロンプト・パラメータを注入してワークフローを準備
        prepared = PromptInjector.inject(
            workflow,
            positive_prompt=positive_prompt or None,
            negative_prompt=negative_prompt or None,
            meta=meta,
            extra_params=extra_params,
        )

        # ComfyUIClient でストリーミング生成
        async for event in self.client.stream_generation(
            workflow=prepared,
            client_id=cid,
            output_type=meta.output_type,
        ):
            yield event
