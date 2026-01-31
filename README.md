# Agent Cognitive Compressor (ACC) Agent

このプロジェクトは、長期的なマルチターン対話における「コンテキストの肥大化」と「目的の忘却（ドリフト）」を解決するための革新的なメカニズムである **Agent Cognitive Compressor (ACC)** を実装した次世代エージェントシステムです。

## 🚀 概要

従来のチャットエージェントは過去の履歴をすべてプロンプトに追加（Transcript Replay）していくため、会話が長くなるほどコストが増大し、重要な情報が埋もれて「目的の忘却」が発生していました。

本プロジェクトでは、**Compressed Cognitive State (CCS)** と呼ばれる、構造化され制限されたサイズの内部状態のみを毎ターン更新（Replacement）して保持します。エージェントは過去の履歴全文を参照する代わりに、この CCS と現在の入力のみを使用して回答を生成します。

## 💡 コア・コンセプト

- **CCS (Compressed Cognitive State)**: 直近の動き、対話の要点、重要なエンティティ、因果関係、目標、制約などを保持する構造化データ。
- **履歴の完全置換 (Replacement)**: 履歴を append するのではなく、毎ターン CCS を上書き更新することでコンテキスト窓の消費を一定に保ちます。
- **忘却と定着**: 重要でない詳細は積極的に捨て、重要な事実は Artifact Store (長期記憶) へ定着させます。

## 🏗️ アーキテクチャ

システムは以下の主要コンポーネントで構成されています。

1.  **ACC Controller**: メモリ更新サイクルを制御する中枢。
2.  **Cognitive Compressor Model (CCM)**: LLMを使用して新しい CCS を生成・更新（圧縮）する担当。
3.  **Agent Engine**: CCSの内容のみを参照して、最終的な回答を生成する推論エンジン。
4.  **Artifact Store (ChromaDB)**: 過去のログや重要な事実を保存する外部ベクトルデータベース。
5.  **Memory Processor**: 「OpenClaw」スタイルを採用し、毎ターンの対話から日記（Journal）の更新や長期記憶の抽出を行います。

### 🔄 処理フロー (Algorithm 1)

1.  **Input**: ユーザーからの入力を受け取る。
2.  **Recall**: 現在の入力に基づき、Artifact Store から関連する過去の記憶を検索。
3.  **Qualify**: 検索された情報が「今の対話に本当に必要か」をLLMが選別。
4.  **Compress & Commit**: 入力、前ターンのCCS、選別された情報を統合し、新しい CCS を生成（上書き）。
5.  **Action**: 更新された CCS のみを参照し、エージェントが回答を**ストリーミング出力**。
6.  **Finalize**: 対話結果から新たな事実を抽出し、Artifact Store や Journal を更新（バックグラウンド処理）。

## ✨ 主な機能

- **リアルタイム CCS 可視化**: Web UI 上で、エージェントの内部状態（思考の変遷）が更新される様子をリアルタイムに確認可能。
- **高速ストリーミング**: `FastAPI` と `LangChain` を活用し、トークン単位でのスムーズな応答を実現。
- **高度なメモリ管理**: 
    - エピソード記憶（直近の対話内容）
    - セマンティック記憶（抽出された事実・知識）
    - 継続的なジャーナリング（振り返り機能）
- **デバッグモード**: `ACC_DEBUG=true` により、プロンプトの各ステップ（Recall/Qualify/Compress）の推論過程をターミナルに表示。

## 🛠️ 技術スタック

- **Core**: Python 3.10+, LangChain, Pydantic
- **Web**: FastAPI, Uvicorn
- **Database**: ChromaDB (Vector DB)
- **Model**: OpenAI (GPT-4o 推奨)
- **Package Manager**: `uv`

## 📦 セットアップ

### インストール

1.  リポジトリをクローンします。
2.  `uv` を介して依存関係をインストールします。
    ```bash
    uv sync
    ```

### 環境設定

`.env` ファイルを作成し、必要な情報を設定してください。

```env
OPENAI_API_KEY=your_api_key_here
ACC_USER_NAME=edom18
ACC_DEBUG=true
```

## 🚀 実行方法

サーバーを起動します。

```bash
uv run python server.py
```

起動後、ブラウザで [http://localhost:8000](http://localhost:8000) にアクセスしてください。

## 📂 プロジェクト構成

- `src/acc_agent/`: コアロジック
    - `core.py`: ACCのメインループとエージェントエンジン
    - `schemas.py`: CCSのPydanticモデル
    - `memory.py`: ChromaDB（Artifact Store）の操作
    - `memory_processor.py`: メモリ抽出・日記更新ロジック
- `agent-settings/`: エージェントの「魂（SOUL）」やユーザー固有の設定
- `static/`: フロントエンド (HTML/JS/CSS)
- `specs/`: 技術仕様書

## 📝 カスタマイズ

`agent-settings/{user_name}/` 配下のファイルを編集することで、エージェントの振る舞いをカスタマイズできます。

- `SOUL.md`: エージェントのパーソナリティ、口調。
- `AGENTS.md`: 動作プロトコル、具体的な指示。
- `USER.md`: ユーザーに関する基本情報。

---
Detailed specifications can be found in `specs/spec.md`.
