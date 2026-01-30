仕様書: Agent Cognitive Compressor (ACC) 実装要件
1. プロジェクト概要
本プロジェクトの目的は、長期的なマルチターン対話において発生する「コンテキストの増大」や「目的の忘却（ドリフト）」を防ぐため、論文  で提案された Agent Cognitive Compressor (ACC) を実装することである。
従来の手法（Transcript Replay：全履歴の追記）を廃止し、「固定サイズの圧縮された認知状態 (Compressed Cognitive State: CCS)」 を毎ターン更新して保持するメカニズムを構築する。
2. アーキテクチャ定義
システムは以下の主要コンポーネントで構成される 。
1. ACC (Controller): メモリ管理を行う主要モジュール。エージェントの推論エンジンとは独立して動作する。
2. CCS (State): ターン間で保持される唯一の永続データ構造（スキーマ定義あり）。
3. CCM (Model): 新しいCCSを生成するためのLLM呼び出し（圧縮・更新担当）。
4. Artifact Store (External Memory): 過去のログやドキュメントを保存する外部データベース（Vector DBなど）。
5. Agent Engine: CCSを参照して最終的な回答やアクションを生成するLLM。
3. データ構造仕様 (CCS Schema)
論文  に基づき、CompressedCognitiveState クラスを Pydantic モデルとして定義する。これは自由記述の要約ではなく、構造化されたデータでなければならない。
Python クラス定義案 (schemas.py)
from pydantic import BaseModel, Field
from typing import List, Optional

class CompressedCognitiveState(BaseModel):
    """
    ACCによって管理される有界な内部状態 (Bounded Internal State)。
    過去の全履歴の代わりに、このオブジェクトのみを次のターンに持ち越す。
    """
    
    # 1. Episodic trace: 直近のターンで何が起きたか（短期記憶）
    episodic_trace: str = Field(..., description="直近の観測事実、ユーザー入力、ツールの結果の簡潔な記録")
    
    # 2. Semantic gist: 現在の対話の要点・意味
    semantic_gist: str = Field(..., description="現在の状況やトピックの抽象的な要約")
    
    # 3. Focal entities: 重要なエンティティ（ID, サーバー名, 固有名詞など）
    focal_entities: List[str] = Field(..., description="現在注目すべき重要なエンティティのリスト（型情報含むとなお良い）")
    
    # 4. Relational map: 因果関係や依存関係
    relational_map: List[str] = Field(default=[], description="イベント間の因果関係や時間的依存関係")
    
    # 5. Goal orientation: 現在の目標（不変）
    goal_orientation: str = Field(..., description="解決すべきタスクの全体的な目標")
    
    # 6. Constraints: 守るべき制約事項（不変・最重要）
    constraints: List[str] = Field(..., description="絶対に違反してはならないルール、禁止事項、ポリシー")
    
    # 7. Predictive cue: 次に予想される展開
    predictive_cue: Optional[str] = Field(None, description="次に実行すべきステップや予想される展開")
    
    # 8. Uncertainty signal: 不確実な情報の明示
    uncertainty_signal: str = Field(..., description="まだ確認されていない事項やリスクレベル")
    
    # 9. Retrieved artifacts: 外部情報の参照（内容はコピーせず参照のみ）
    retrieved_artifacts: List[str] = Field(default=[], description="判断の根拠となった外部情報の参照IDや要点")

4. 処理フローとアルゴリズム
ACCのメインループは以下の手順（Algorithm 1 ）に従って実装する。
処理ステップ
1. Input (x_t): ユーザーからの新しい入力、またはツールの実行結果を受け取る。
2. Recall (A_t):
    - 現在の入力 x_t と 前の内部状態 CCS_{t-1} をクエリとして、外部ストア（Artifact Store）から関連情報を検索する。
3. Qualify (A_t^+ ):
    - 検索された情報 (A_t) をフィルタリングし、現在の意思決定に真に必要な情報 (A_t^+​) だけを選別する。
4. Compress & Commit (CCS_t​):
    - 入力: x_t​ （現在入力）, CCS_{t−1}​ （前状態）, A_t^+ ​ （選別された外部情報）
    - 処理: CCM（LLM）を使用し、上記の情報を統合して新しい CCS_t​ を生成する。
    - 制約: 必ず上記 CompressedCognitiveState のスキーマに従って出力させる。
    - 更新: 旧状態 CCS_{t−1}​ を破棄し、CCS_t​ に完全に置き換える（Replace）。
5. Action:
    - エージェント本体は、Prompt(System + CCS_t + x_t) の情報のみを使って回答を生成またはツールを実行する（過去の履歴全文は見せない）。

5. モジュール実装要件
A. ACC Controller クラス
• 役割: メモリ更新サイクルの制御。
• メソッド:
    - update_state(current_input, prev_ccs) -> new_ccs: 上記フローの1〜4を実行。
    - 内部でLLM（CCM）を呼び出す際は、「要約」ではなく「状態更新」を行うようプロンプトエンジニアリングが必要。
B. Cognitive Compressor Model (CCM) のプロンプト
ACC専用のプロンプトを用意すること。以下のような指示を含める必要がある。
「あなたはエージェントの認知管理者です。ユーザーとの会話履歴をそのまま保存するのではなく、意思決定に必要な『状態』だけを更新してください。 特に『Constraints（制約）』と『Goal（目標）』は、一度設定されたら明示的に変更がない限り維持し続けてください。 重要でない詳細は積極的に捨て（忘却し）、スキーマの各フィールドを最新の事実に書き換えてください。」
C. エージェント統合 (ReAct Loop) 
既存のLangChain等のReActエージェントに組み込む場合の変更点：
- Before: Memory = [Message(t-1), Message(t-2), ...] (リスト追記型)
- After: Context = CCS_t (オブジェクト置換型)
エージェントのシステムプロンプトには、常に最新の CCS_t の内容（JSONまたはフォーマットされたテキスト）が挿入されるようにする。

6. 実装上の注意点
1. 履歴の肥大化防止: 実装コード内で、リストにメッセージを append する処理がないか厳重にチェックする。ACCは「Replacement（置き換え）」が原則である。
2. 外部知識の取り扱い: RAG（検索）の結果をそのままプロンプトに入れないこと。一度ACCを通して「この検索結果は現在のゴールに関連するか？」を判断させ、関連する場合のみ CCS の retrieved_artifacts や episodic_trace に反映させる。
3. 不変条件の維持: 「営業時間内は再起動禁止」のような制約（Constraints）が、ターンが進んでも消えないことをテストケースで確認する。

7. 推奨ライブラリ構成
- Python: 3.10+
- Pydantic: スキーマ定義とバリデーション用
- LangChain / LangGraph: エージェントループとツール実行の管理
- OpenAI API (または他のLLM): CCMおよびエージェントのバックエンド
- ChromaDB / FAISS: Artifact Store用（オプション、長期記憶が必要な場合）