import os
from pathlib import Path
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import datetime

from .schemas import CompressedCognitiveState

class IntrospectionAgent:
    """
    自律的な更新を行うサブエージェント。
    会話から「記憶すべき事実」と「自身の設定変更」を検知し、実行する。
    """
    def __init__(self, user_name: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.user_name = user_name
        self.settings_dir = Path(f"agent-settings/{self.user_name}")
        
    def _read_file(self, filename: str) -> str:
        file_path = self.settings_dir / filename
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return ""

    def _write_file(self, filename: str, content: str):
        file_path = self.settings_dir / filename
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        if os.getenv("ACC_DEBUG", "false").lower() == "true":
            print(f"[Introspection] Updated {filename}")

    def run_introspection_cycle(self, user_input: str, agent_response: str, ccs: CompressedCognitiveState) -> dict:
        """
        内省サイクルを実行する。
        1. 日記の作成 (Daily Journal)
        2. 事実の抽出 (Memory Extraction)
        3. コンテキストの更新チェック (Context Update)
        """
        results = {
            "journal_entry": "",
            "facts": [],
            "updated_files": []
        }
        
        # 1. Daily Journal
        results["journal_entry"] = self.create_daily_journal_entry(user_input, agent_response)
        
        # 2. Extract Facts
        results["facts"] = self.extract_memories(user_input, agent_response, ccs)
        
        # 3. Check for Context Updates
        updated_files = self.check_and_update_context(user_input, agent_response)
        results["updated_files"] = updated_files
        
        return results

    def check_and_update_context(self, user_input: str, agent_response: str) -> List[str]:
        """
        会話内容から、USER.md や AGENTS.md の更新が必要か判断し、必要なら更新する。
        """
        current_user_md = self._read_file("USER.md")
        current_agents_md = self._read_file("AGENTS.md")
        
        system_prompt = """
あなたはAIエージェントの「設定管理者」です。
ユーザーとの直近の会話内容に基づいて、以下の2つの設定ファイルを更新する必要があるか判断してください。

1. **USER.md**: ユーザーのプロフィール情報 (名前、役割、好み、現状など)
2. **AGENTS.md**: エージェント自身の振る舞いルール (話し方、制約、守るべき指針)

# 更新判断基準
- ユーザーから**明示的な変更依頼**があった場合 (例:「今後は敬語をやめて」「転職してCTOになった」)
- ユーザーに関する**重要な事実の変化**が確実な場合
- 新しい**恒久的なルール**が追加された場合

# 現在のファイル内容

## USER.md
{current_user_md}

## AGENTS.md
{current_agents_md}

# 会話内容
User: {user_input}
AI: {agent_response}

# 指示
更新が必要な場合のみ、更新後のファイル内容全体を作成してください。
変更がない場合は、対象フィールドを空文字またはnullにしてください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        
        class ContextUpdate(BaseModel):
            new_user_md: Optional[str] = Field(description="更新後のUSER.mdの内容。変更なしならNone")
            new_agents_md: Optional[str] = Field(description="更新後のAGENTS.mdの内容。変更なしならNone")
            reason: str = Field(description="更新理由。更新しない場合はその理由")

        chain = prompt | self.llm.with_structured_output(ContextUpdate)
        
        try:
            result = chain.invoke({
                "current_user_md": current_user_md,
                "current_agents_md": current_agents_md,
                "user_input": user_input,
                "agent_response": agent_response
            })
            
            updated_files = []
            
            if result.new_user_md and result.new_user_md.strip() != current_user_md.strip():
                # Safety check: Don't allow empty wipes unless explicit? 
                # Assuming prompt handles this mostly, but let's be safe.
                if len(result.new_user_md) > 10: 
                    self._write_file("USER.md", result.new_user_md)
                    updated_files.append("USER.md")
            
            if result.new_agents_md and result.new_agents_md.strip() != current_agents_md.strip():
                if len(result.new_agents_md) > 10:
                    self._write_file("AGENTS.md", result.new_agents_md)
                    updated_files.append("AGENTS.md")
            
            if updated_files and os.getenv("ACC_DEBUG", "false").lower() == "true":
                print(f"[Introspection] Context Update Reason: {result.reason}")
                
            return updated_files

        except Exception as e:
            print(f"Error in check_and_update_context: {e}")
            return []

    def extract_memories(self, user_input: str, agent_response: str, ccs: CompressedCognitiveState) -> List[str]:
        """
        現在のターンから、将来の意思決定に役立つ重要事実があるかを判定し、抽出する。
        (MemoryProcessorから移植・統合)
        """
        system_prompt = """
あなたはエージェントの記憶選別官です。
以下の対話から「長期的に記憶すべき事実（Facts）」を抽出してください。
これらの事実は `MEMORY.md` に永続化され、将来のセッションで参照されます。

# 抽出基準
1. **ユーザーの属性・自己紹介**: 名前、職業、役割、趣味など。
2. **ユーザーの好み**: 好きな言語、ツール、嫌いなものなど。
3. **プロジェクトの決定事項**: 技術選定、設計方針など。
4. **重要な制約**: 「XXXは使うな」などの禁止事項。

# 除外基準
1. 挨拶や社交辞令。
2. 天気の話など、その場限りの話題。
3. **「USER.md」や「AGENTS.md」を更新するほどの内容** (これは別途処理されるため、重複を避ける)。
   ただし、具体的な事実 (例:「AWSを使うことにした」) は記憶としても残して良い。

# 入力情報
- 現在の入力: {user_input}
- エージェントの応答: {agent_response}
- 直近の文脈 (CCS): {ccs_gist}

指示：
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
                "ccs_gist": ccs.semantic_gist if ccs else ""
            })
            return result.facts
        except Exception as e:
            print(f"Error in extract_memories: {e}")
            return []

    def create_daily_journal_entry(self, user_input: str, agent_response: str) -> str:
        """
        今日の日記（Journal）への追記エントリを作成する。
        """
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        
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

# フォーマット
記録する場合は、以下のMarkdown形式（日本語）で出力してください。
```markdown
## {current_time} - [タイトル]
[内容を簡潔に記述]
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
