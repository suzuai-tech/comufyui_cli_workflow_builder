# ComfyUI CLI Workflow Builder 使い方

このツールは、ComfyUI のワークフローを使って CLI から画像/動画を生成するためのものです。

## 1. 事前準備

1. ComfyUI を起動する
2. 必要モデル/カスタムノードを ComfyUI 側に配置する
3. 必要なら環境変数で接続先を設定する

- `COMFYUI_HOST`（既定: `192.168.10.35`）
- `COMFYUI_PORT`（既定: `8188`）
- `COMFYUI_API_URL`（未指定時は host/port から自動生成）

## 2. 実行コマンド

プロジェクトルートで実行します。

```bash
python comfyui_cli.py <subcommand> [options]
```

（従来の `python -m core.generate_image_core ...` でも実行可能です）

利用可能サブコマンド:

- `list` : 利用可能ワークフロー一覧
- `convert` : ComfyUI からDLした `workflow.json` を保存/変換
- `generate` : 生成実行
- `interrupt` : 実行中ジョブを中断
- `clear-queue` : キューを全クリア
- `queue-info` : キュー状態を表示

---

## 3. list

```bash
python comfyui_cli.py list
```

出力例:

- workflow id
- name
- type(image/video)

---

## 4. convert

ComfyUI の JSON（UI形式 / API形式どちらでも可）を登録します。

```bash
python comfyui_cli.py convert \
  --input "C:/path/to/workflow.json" \
  --workflow-id my_workflow \
  --name "My Workflow"
```

主なオプション:

- `--input` 必須: 変換元JSON
- `--workflow-id` 必須: 保存先ID（`comfyui_workflows/<id>/`）
- `--name` 任意: 表示名
- `--description` 任意: 説明
- `--output-type` 任意: `image` / `video`（既定: `image`）
- `--output-format` 任意: 既定 `png`

### convert時の注意

- 自動検出されるのは主に `positive_node_id` / `negative_node_id` です。
- `param_map` は自動生成されません（空で保存）。
  - 必要なら `comfyui_workflows/<id>/meta.json` を手で編集してください。

---

## 5. generate

```bash
python comfyui_cli.py generate \
  --workflow-id txt2img_basic \
  --prompt "a beautiful landscape"
```

主なオプション:

- `--prompt` 必須: ポジティブプロンプト
- `--negative` 任意: ネガティブプロンプト（未指定時はデフォルト）
- `--workflow-id` 任意: 使うワークフローID（既定: `txt2img_basic`）
- `--param` 任意: 追加パラメータ（複数指定可）

### `--param` の指定方法

以下の3パターンに対応しています。

1. `meta.json` の `param_map` キー名で指定
2. `node_id:field=value` で直接指定
3. KSampler の既存入力名（例: `seed`, `steps`, `cfg`）を自動注入

例:

```bash
python comfyui_cli.py generate \
  --workflow-id txt2img_zimage \
  --prompt "portrait photo" \
  --param seed=0 \
  --param steps=20 \
  --param cfg=5.5
```

直接指定の例:

```bash
python comfyui_cli.py generate \
  --workflow-id txt2img_basic \
  --prompt "portrait photo" \
  --param 3:steps=30
```

### `--param` 未指定時

- ワークフロー JSON に元々含まれる値が使用されます。
- `--prompt` は必須なので、ポジティブテキストは指定値で上書きされます。

---

## 6. キュー操作

### 実行中ジョブの中断

全体中断:

```bash
python comfyui_cli.py interrupt
```

特定 `client_id` のみ中断:

```bash
python comfyui_cli.py interrupt --client-id "<client_id>"
```

### キュー全クリア

```bash
python comfyui_cli.py clear-queue
```

### キュー状態確認

```bash
python comfyui_cli.py queue-info
```

---

