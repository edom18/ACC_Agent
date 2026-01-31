from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from .schemas import CompressedCognitiveState
import os

class MemoryProcessor:
    """
    対話から長期記憶（Facts）を抽出する専用プロセッサ。
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)

    def extract_memories(self, user_input: str, agent_response: str, ccs: CompressedCognitiveState) -> List[str]:
        """
        現在のターンから、将来の意思決定に役立つ重要事実があるかを判定し、抽出する。
        """
        system_prompt = """
あなたはエージェントの記憶選別官です。
以下の対話から「長期的に記憶すべき事実（Facts）」を抽出してください。
これらの事実は `MEMORY.md` に永続化され、将来のセッションで参照されます。

# 抽出基準
1. **ユーザーの属性・自己紹介**: 名前、職業、役割、趣味など（例:「私はJackだ」→「ユーザーの名前はJack」）。
2. **ユーザーの好み**: 好きな言語、ツール、嫌いなものなど（例:「Pythonが好き」→「ユーザーはPythonが好き」）。
3. **プロジェクトの決定事項**: 技術選定、ポート番号、設計方針など。
4. **重要な制約**: 「XXXは使うな」などの禁止事項。

# 除外基準
1. 挨拶や社交辞令（「こんにちは」「ありがとう」）。
2. 天気の話など、その場限りの話題。
3. エージェント自身の振る舞いに関するフィードバック（これは別途保存されるため）。

# 入力情報
- 現在の入力: {user_input}
- エージェントの応答: {agent_response}
- 直近の文脈 (CCS): {ccs_gist}

指示：
- 文脈(CCS)にある情報でも、永続化（長期記憶）する価値がある明確な事実は抽出してください。
- 事実は簡潔な箇条書きのテキストで出力してください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])

        class MemoryExtraction(BaseModel):
            facts: List[str] = Field(default=[], description="長期的に保存すべき事実のリスト")

        chain = prompt | self.llm.with_structured_output(MemoryExtraction)

        try:
            result = chain.invoke({
                "user_input": user_input,
                "agent_response": agent_response,
                "ccs_gist": ccs.semantic_gist
            })
            
            # デバッグログ
            if os.getenv("ACC_DEBUG", "false").lower() == "true" and result.facts:
                print(f"\n[MemoryProcessor] Extracted Facts: {result.facts}\n")
                
            return result.facts
        except Exception as e:
            print(f"Error in extract_memories: {e}")
            return []

    def update_daily_journal(self, current_journal: str, user_input: str, agent_response: str) -> str:
        """
        今日の日記（Journal）を更新する。
        新しい対話内容を踏まえて、既存の日記をリライト（再構成）する。
        """
        system_prompt = """
あなたはAIエージェント自身です。
今日一日の出来事を記録する「業務日誌（Journal）」を書いています。

# 入力情報
- **現在の業務日誌**: {current_journal}
- **最新のユーザー入力**: {user_input}
- **最新のあなたの応答**: {agent_response}

# 指示
既存の日記の内容を保持しつつ、最新のやり取りを自然に統合して、日記全体を書き直してください。
- **視点**: 一人称（私 / I）。
- **スタイル**: 客観的かつ簡潔に。箇条書きではなく、自然な文章（ナラティブ）で記述する。
- **内容**: ユーザーが何に取り組み、どのような問題を解決したか、どのような決定をしたかを記録する。
- **追記ではなく統合**: 単に末尾に追加するのではなく、文脈がつながるように全体を調整してよい。日付が変わっていなければ、一つのまとまりのある文章にする。

出力は、更新された日記の本文のみを返してください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        
        class JournalUpdate(BaseModel):
            journal_content: str = Field(..., description="更新された日記の本文")
            
        chain = prompt | self.llm.with_structured_output(JournalUpdate)
        
        try:
            result = chain.invoke({
                "current_journal": current_journal if current_journal else "(まだ記録なし)",
                "user_input": user_input,
                "agent_response": agent_response
            })
            return result.journal_content
        except Exception as e:
            print(f"Error in update_daily_journal: {e}")
            # エラー時は既存＋今回の追記（バックアップ的処理）
            return f"{current_journal}\n\n*Update Error: {e}*\nUser: {user_input}\nAgent: {agent_response}"
