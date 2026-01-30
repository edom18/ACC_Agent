# Agent Cognitive Compressor (ACC) Agent

このプロジェクトは、長期的なマルチターン対話における「コンテキストの肥大化」と「目的の忘却（ドリフト）」を解決するためのメカニズムである **Agent Cognitive Compressor (ACC)** を実装したエージェントシステムです。

## 概要

従来のチャットエージェントは過去の履歴をすべてプロンプトに追加していきましたが、本プロジェクトでは **Compressed Cognitive State (CCS)** と呼ばれる、構造化され、制限されたサイズの内部状態のみを毎ターン更新して保持します。

エージェントは過去の履歴全文を参照する代わりに、この CCS と現在の入力のみを使用して回答を生成します。これにより、トークン消費の抑制と、長期的な目標の維持を両立します。

### 主な特徴
- **CCS (Compressed Cognitive State)**: 直近の動き、対話の要点、重要なエンティティ、因果関係、目標、制約などを構造化データとして保持。
- **逐次更新 (Replacement)**: 履歴を追記するのではなく、毎ターン CCS を上書き更新。
- **LangChain 連携**: LangChain を使用した LLM 操作と状態管理。
- **Web UI**: エージェントの状態（CCS）がリアルタイムに可視化されるインタフェース。

## プロジェクト構成

- `src/acc_agent/`: コアロジック。
  - `schemas.py`: CCS (Compressed Cognitive State) の Pydantic モデル定義。
  - `core.py`: CCS を更新し、回答を生成する ACC コントローラー。
- `server.py`: FastAPI を使用した Web サーバー。
- `static/`: フロントエンドの HTML / JS / CSS。
- `specs/`: プロジェクトの技術仕様書。

## セットアップ

### 前提条件
- Python 3.10 以上
- `uv` (Python パッケージマネージャー)

### インストール

1. リポジトリをクローンします。
2. 依存関係をインストールします。
   ```bash
   uv sync
   ```

### 環境設定

`.env` ファイルを作成し、OpenAI API キーを設定してください。

```env
OPENAI_API_KEY=your_api_key_here
```

## 実行方法

サーバーを起動します。

```bash
uv run python server.py
```

起動後、ブラウザで [http://localhost:8000](http://localhost:8000) にアクセスしてください。

## 使い方

1. チャット画面からエージェントにメッセージを送信します。
2. エージェントの回答とともに、右側のパネルで **Cognitive State (CCS)** がどのように更新されたかを確認できます。
3. 会話が進んでも、エージェントが「目標」や「制約」を維持していること、履歴が肥大化していないことを確認してください。

## 技術詳細

詳細な仕様については `specs/spec.md` を参照してください。
