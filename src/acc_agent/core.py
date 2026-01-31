import os
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from .schemas import CompressedCognitiveState
from .memory import ArtifactStore
from .memory_manager import MemoryManager
from .memory_processor import MemoryProcessor

def _log_llm_interaction(step_name: str, prompt: Any, response: Any):
    if os.getenv("ACC_DEBUG", "false").lower() != "true":
        return
    
    separator = "=" * 50
    print(f"\n{separator}")
    print(f"DEBUG: {step_name}")
    print(f"{separator}")
    print("--- PROMPT ---")
    if hasattr(prompt, "messages"):
        for msg in prompt.messages:
            print(f"[{msg.type}]: {msg.content}")
    else:
        print(prompt)
    
    print("\n--- RESPONSE ---")
    if hasattr(response, "model_dump_json"):
        print(response.model_dump_json(indent=2))
    else:
        print(response)
    print(f"{separator}\n")

class CognitiveCompressorModel:
    """
    Cognitive Compressor Model (CCM)
    CCSの更新とアーティファクトの選別を担当する。
    """
    def __init__(self, agents_context: str = "", model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.agents_context = agents_context

    def qualify_artifacts(self, current_input: str, prev_ccs: Optional[CompressedCognitiveState], artifacts: list[str]) -> list[str]:
        """
        Qualify (Step 3): 検索された情報をフィルタリングし、真に必要なものだけを選別する。
        """
        if not artifacts:
            return []
        
        system_prompt = """
あなたは情報の精査官です。
提供された「外部想起情報」の中から、現在の対話文脈において「意思決定や正確な回答に不可欠な情報」だけを抽出してください。
少しでも関連が薄い、あるいは現在のCCSや入力から推測可能な情報は除外してください。

# 前回の状態 (Previous State)
{prev_state_json}

# 現在の入力 (Current Input)
{current_input}

# 外部想起情報 (Retrieved Artifacts)
{artifacts_list}

指示：
- 選別された情報のリストをJSON形式で返してください。
- 該当する情報がない場合は空のリストを返してください。
"""
        prev_state_json = prev_ccs.model_dump_json(indent=2) if prev_ccs else "（なし）"
        artifacts_list = "\n".join([f"- {a}" for a in artifacts])

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        
        # Simple list of strings output
        class SelectedArtifacts(BaseModel):
            selected: list[str] = Field(..., description="選別された情報のリスト")

        chain = prompt | self.llm.with_structured_output(SelectedArtifacts)
        
        input_vars = {
            "prev_state_json": prev_state_json,
            "current_input": current_input,
            "artifacts_list": artifacts_list
        }
        
        try:
            result = chain.invoke(input_vars)
            # Log the interaction
            _log_llm_interaction("STEP 3: Qualify Artifacts", prompt.format_messages(**input_vars), result)
            return result.selected
        except Exception as e:
            if os.getenv("ACC_DEBUG", "false").lower() == "true":
                print(f"DEBUG: Qualify Artifacts Failed: {e}")
            return []

    def compress_and_commit(self, current_input: str, prev_ccs: Optional[CompressedCognitiveState], qualified_artifacts: list[str], long_term_memory: str = "") -> CompressedCognitiveState:
        """
        Compress & Commit (Step 4): 情報を統合して新しいCCSを生成する。
        """
        system_prompt = """
あなたはエージェントの認知管理者 (Cognitive Manager) です。
ユーザーとの会話履歴をそのまま保存するのではなく、意思決定に必要な「状態 (State)」だけを更新してください。

# 動作ルール (Agents Protocols)
{agents_context}

# 既存の長期記憶 (Existing Long-term Knowledge)
{long_term_memory}

指示：
長期記憶に既に存在する情報は、CCSに重複して保存しないでください。



# 前回の状態 (Previous State)
{prev_state_json}

# 選別された外部情報 (Qualified Artifacts)
{artifacts}

# 現在の入力 (Current Input)
{current_input}

# 指示
今の入力と前回の状態、および外部情報を統合し、新しい「圧縮された認知状態 (Compressed Cognitive State)」を生成してください。
特に以下の点に注意してください：
1. **Constraints (制約)** と **Goal (目標)** は、一度確立されたら明示的に変更・完了の指示がない限り維持し続けてください（不変項目の維持）。
2. 重要でない詳細は積極的に捨て（忘却し）、スキーマの各フィールドを最新の事実に書き換えてください。
3. `episodic_trace` は直近の出来事を簡潔に記述してください。
4. `semantic_gist` は全体の流れを要約してください。
5. `retrieved_artifacts` には今回参照した外部情報の要点を記録してください。
"""
        prev_state_json = prev_ccs.model_dump_json(indent=2) if prev_ccs else "（なし：初回起動）"
        artifacts_str = "\n".join(qualified_artifacts) if qualified_artifacts else "（なし）"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        
        chain = prompt | self.llm.with_structured_output(CompressedCognitiveState)
        
        input_vars = {
            "prev_state_json": prev_state_json,
            "artifacts": artifacts_str,
            "current_input": current_input,
            "agents_context": self.agents_context,
            "long_term_memory": long_term_memory
        }
        
        new_ccs = chain.invoke(input_vars)
        
        # Log the interaction
        _log_llm_interaction("STEP 4: Compress & Commit", prompt.format_messages(**input_vars), new_ccs)
        
        return new_ccs

class AgentEngine:
    """
    CCSを参照して最終的な回答を生成するエージェント本体。
    履歴全文は見ず、CCSと現在の入力のみを見る。
    """
    def __init__(self, soul_context: str = "", user_context: str = "", agents_context: str = "", model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.soul_context = soul_context
        self.user_context = user_context
        self.agents_context = agents_context

    def generate_response(self, current_input: str, ccs: CompressedCognitiveState) -> str:
        system_prompt = """
{soul_context}

{user_context}

{agents_context}

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
        
        input_vars = {
            "ccs_json": ccs.model_dump_json(indent=2),
            "current_input": current_input,
            "soul_context": self.soul_context,
            "user_context": self.user_context,
            "agents_context": self.agents_context
        }
        
        response = chain.invoke(input_vars)
        
        # Log the interaction
        _log_llm_interaction("STEP 5: Action (Agent Response)", prompt.format_messages(**input_vars), response.content)
        
        return response.content

class ACCController:
    """
    ACCのメインコントローラー。
    メモリ更新サイクルを制御する。
    """
    def __init__(self):
        self.user_name = os.getenv("ACC_USER_NAME", "edom18")
        self.settings_dir = Path(f"agent-settings/{self.user_name}")
        
        self.soul_context = self._load_context_file("SOUL.md")
        self.user_context = self._load_context_file("USER.md")
        self.agents_context = self._load_context_file("AGENTS.md")

        # Initialize Memory Components
        self.memory_manager = MemoryManager(user_name=self.user_name)
        self.memory_processor = MemoryProcessor()

        self.ccm = CognitiveCompressorModel(agents_context=self.agents_context)
        self.agent = AgentEngine(
            soul_context=self.soul_context,
            user_context=self.user_context,
            agents_context=self.agents_context
        )
        self.store = ArtifactStore()
        self.current_ccs: Optional[CompressedCognitiveState] = None

    def _load_context_file(self, filename: str) -> str:
        try:
            path = self.settings_dir / filename
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")
        return ""

    def process_turn(self, user_input: str) -> Dict[str, Any]:
        """
        ACCメインループ (Algorithm 1):
        1. Input
        2. Recall
        3. Qualify
        4. Compress & Commit
        5. Action
        """
        
        # 1. Recall (Step 2)
        # 入力と現在の状態（要約）の両方をクエリとして使用
        recall_query = f"{user_input}\nContext: {self.current_ccs.semantic_gist if self.current_ccs else ''}"
        raw_artifacts = self.store.recall(recall_query)
        
        # 2. Qualify (Step 3)
        qualified_artifacts = self.ccm.qualify_artifacts(user_input, self.current_ccs, raw_artifacts)
        
        # Load Long-term Memory for CCM
        ltm_content = self.memory_manager.read_long_term_memory()

        # 3. Compress & Commit (Step 4)
        new_ccs = self.ccm.compress_and_commit(
            user_input, 
            self.current_ccs, 
            qualified_artifacts,
            long_term_memory=ltm_content
        )
        
        # Update internal state (Replacement)
        self.current_ccs = new_ccs
        
        # 4. Action (Step 5)
        response_text = self.agent.generate_response(user_input, self.current_ccs)
        
        # --- Memory Updates (OpenClaw Style) ---
        
        # 1. Daily Log
        self.memory_manager.append_to_daily_log(user_input, response_text)
        
        # 2. Memory Flush (Extract Facts)
        facts = self.memory_processor.extract_memories(user_input, response_text, new_ccs)
        if facts:
            self.memory_manager.append_to_long_term_memory(facts)
            # Also add to vector store for retrieval
            for fact in facts:
                self.store.add_artifact(fact, metadata={"type": "semantic_memory", "source": "memory_flush"})

        # (Legacy) Episodic Trace for Artifact Store
        # 今回のCCSのコピーを保存

        self.store.add_artifact(
            content=f"User: {user_input}\nAssistant: {response_text}\nGist: {new_ccs.semantic_gist}",
            metadata={"type": "episodic_memory"}
        )
        
        return {
            "response": response_text,
            "ccs": self.current_ccs.model_dump()
        }
