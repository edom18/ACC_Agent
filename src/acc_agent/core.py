import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from .schemas import CompressedCognitiveState

class CognitiveCompressorModel:
    """
    负责更新 CompressedCognitiveState (CCS) 的模块。
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        
    def compress_and_commit(self, current_input: str, prev_ccs: Optional[CompressedCognitiveState], retrieved_artifacts: list[str]) -> CompressedCognitiveState:
        
        system_prompt = """
あなたはエージェントの認知管理者 (Cognitive Manager) です。
ユーザーとの会話履歴をそのまま保存するのではなく、意思決定に必要な「状態 (State)」だけを更新してください。

# 前回の状態 (Previous State)
{prev_state_json}

# 外部想起情報 (Retrieved Artifacts)
{artifacts}

# 現在の入力 (Current Input)
{current_input}

# 指示
今の入力と前回の状態を統合し、新しい「圧縮された認知状態 (Compressed Cognitive State)」を生成してください。
特に以下の点に注意してください：
1. **Constraints (制約)** と **Goal (目標)** は、一度設定されたら明示的に変更指示がない限り維持し続けてください（不変項目の維持）。
2. 重要でない詳細は積極的に捨て（忘却し）、スキーマの各フィールドを最新の事実に書き換えてください。
3. `episodic_trace` は直近の出来事を簡潔に記述してください。
4. `semantic_gist` は全体の流れを要約してください。
"""
        # If it's the first turn, prev_state is None or empty.
        prev_state_json = prev_ccs.model_dump_json(indent=2) if prev_ccs else "（なし：初回起動）"
        artifacts_str = "\n".join(retrieved_artifacts) if retrieved_artifacts else "（なし）"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        
        # Use structured output to ensure we get the CCS schema back
        chain = prompt | self.llm.with_structured_output(CompressedCognitiveState)
        
        new_ccs = chain.invoke({
            "prev_state_json": prev_state_json,
            "artifacts": artifacts_str,
            "current_input": current_input
        })
        return new_ccs

class AgentEngine:
    """
    CCSを参照して最終的な回答を生成するエージェント本体。
    履歴全文は見ず、CCSと現在の入力のみを見る。
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)

    def generate_response(self, current_input: str, ccs: CompressedCognitiveState) -> str:
        system_prompt = """
あなたはAIアシスタントです。
以下の「圧縮された認知状態 (Compressed Cognitive State)」のみをコンテキストとして持ち、ユーザーに応答してください。
過去の会話履歴の生データはありません。この重要事項の要約（CCS）だけが全てです。

# 現在の認知状態 (Current Cognitive State)
{ccs_json}

この状態に基づき、ユーザーの入力に対して適切に応答・アクションを行ってください。
制約事項 (Constraints) は必ず守ってください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{current_input}")
        ])
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "ccs_json": ccs.model_dump_json(indent=2),
            "current_input": current_input
        })
        
        return response.content

class ACCController:
    """
    ACCのメインコントローラー。
    メモリ更新サイクルを制御する。
    """
    def __init__(self):
        self.ccm = CognitiveCompressorModel()
        self.agent = AgentEngine()
        # In a real app, this might be persisted in a DB keyed by session_id
        self.current_ccs: Optional[CompressedCognitiveState] = None

    def process_turn(self, user_input: str) -> Dict[str, Any]:
        """
        1ターン分の処理を実行する。
        Input -> Recall(mock) -> Compress -> Update State -> Action
        """
        
        # 1. Recall (Mocked for now)
        # 実際にはここで Vector DB から検索を行う
        retrieved_artifacts = [] 
        
        # 2. Compress & Commit (State Update)
        # 前の状態 + 入力 -> 新しい状態
        new_ccs = self.ccm.compress_and_commit(user_input, self.current_ccs, retrieved_artifacts)
        
        # Update internal state (Replacement)
        self.current_ccs = new_ccs
        
        # 3. Action
        # 新しい状態 + 入力 -> 回答生成
        response_text = self.agent.generate_response(user_input, self.current_ccs)
        
        return {
            "response": response_text,
            "ccs": self.current_ccs.model_dump()
        }
