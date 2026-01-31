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

    def create_daily_journal_entry(self, user_input: str, agent_response: str) -> str:
        """
        今日の日記（Journal）への追記エントリを作成する。
        """
        import datetime
        current_time = datetime.datetime.now().strftime("%I:%M %p") # 02:30 PM format
        
        system_prompt = """
あなたはAIエージェント自身です。
ユーザーとのやり取りを記録する「日記（Daily Journal）」を書いています。

# 入力情報
- **現在時刻**: {current_time}
- **ユーザー入力**: {user_input}
- **あなたの応答**: {agent_response}

# 記録判断基準
以下のいずれかの場合のみ記録を行ってください。それ以外は空文字を返してください。
1. エージェントとして覚えておくべき重要な決定事項や進捗があった場合。
2. ユーザーから「覚えておいて」と明示的に言われた場合。
3. プロジェクトの仕様変更や重要な事実が判明した場合。

（挨拶、雑談、すでに記録済みの内容の繰り返しは記録不要です）

# フォーマット
記録する場合は、以下のMarkdown形式（日本語）で出力してください。
タイトルは具体的かつ簡潔にしてください。

```markdown
## {current_time} - [タイトル]
[内容を簡潔に記述]
```

例:
```markdown
## 10:30 AM - API設計の議論
ユーザーとREST vs GraphQLについて議論。シンプルさを重視してRESTを採用することに決定。
主要エンドポイント: /users, /auth, /projects
```

出力は、追記すべきテキスト（Markdown）のみを返してください。記録不要な場合は "NONE" とだけ返してください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        
        class JournalEntry(BaseModel):
            content: str = Field(..., description="追記するエントリ内容。記録不要ならNONE")
            
        chain = prompt | self.llm.with_structured_output(JournalEntry)
        
        try:
            result = chain.invoke({
                "current_time": current_time,
                "user_input": user_input,
                "agent_response": agent_response
            })
            
            if result.content.strip() == "NONE":
                return ""
            
            return result.content
        except Exception as e:
            print(f"Error in create_daily_journal_entry: {e}")
            return ""
