import os
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, AsyncIterator
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
        
    print(f"\n--- {step_name.upper()} ---")
    print("--- PROMPT ---")
    if isinstance(prompt, list):
        for msg in prompt:
            print(f"{msg.type.upper()}: {msg.content}")
    else:
        print(prompt)
    print("--- RESPONSE ---")
    print(response)
    print("==================================================")

class CognitiveCompressorModel:
    """
    認知圧縮モデル (CCM)。
    短期記憶(CCS)の更新と、長期記憶(LTM)へのインタフェースを担う。
    Implementation based on: "The Cognitive Compressor: Optimized for bounded context windows"
    """
    def __init__(self, agents_context: str = "", model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.agents_context = agents_context

    def qualify_artifacts(self, current_input: str, ccs: Optional[CompressedCognitiveState], artifacts: list[str]) -> list[str]:
        """
        Qualify (Step 3): Recallされた情報（Artifacts）の関連性を評価し、フィルタリングする。
        """
        if not artifacts:
            return []
            
        system_prompt = """
あなたはエージェントの記憶選別官です。
ユーザーの入力と現在の状態に基づき、検索された過去の記憶（Artifacts）が「今の対話に必要かどうか」を判定してください。

# 現在の入力
{current_input}

# 現在の状態要約
{ccs_gist}

# 判定基準
- 現在のタスクや質問に直接関連する情報か？
- 文脈を補完するために不可欠か？

必要なArtifactのみをリストとして返してください。不要な場合は空リストを返してください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Artifacts: {artifacts_list}")
        ])
        
        class QualifiedList(BaseModel):
            selected: list[str] = Field(description="関連性が高いと判断されたArtifactの内容リスト")

        chain = prompt | self.llm.with_structured_output(QualifiedList)
        
        try:
            result = chain.invoke({
                "current_input": current_input,
                "ccs_gist": ccs.semantic_gist if ccs else "None",
                "artifacts_list": "\n---\n".join(artifacts)
            })
            
            _log_llm_interaction("STEP 3: Qualify Artifacts", prompt.format_messages(current_input=current_input, ccs_gist=ccs.semantic_gist if ccs else "None", artifacts_list="\n---\n".join(artifacts)), result.selected)
            
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

# 関連する過去の記憶 (Qualified Artifacts)
{artifacts}

# 新しい入力 (Current Input)
{current_input}

これらを統合し、新しい「圧縮された認知状態 (CCS)」を出力してください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
        ])
        
        chain = prompt | self.llm.with_structured_output(CompressedCognitiveState)
        
        prev_state_json = prev_ccs.model_dump_json(indent=2) if prev_ccs else "None (Initial State)"
        artifacts_str = "\n---\n".join(qualified_artifacts) if qualified_artifacts else "None"
        
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
        # (Sync version kept for legacy/testing if needed, or could just wrap async)
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
        _log_llm_interaction("STEP 5: Action (Agent Response)", prompt.format_messages(**input_vars), response.content)
        return response.content

    async def generate_response_stream(self, current_input: str, ccs: CompressedCognitiveState) -> AsyncIterator[str]:
        """
        ストリーミングレスポンスを生成する非同期ジェネレータ。
        """
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
        
        # Log prompt
        _log_llm_interaction("STEP 5: Action (Stream Start)", prompt.format_messages(**input_vars), "(Streaming...)")

        # Use astream for async streaming
        async for chunk in chain.astream(input_vars):
            if chunk.content:
                yield chunk.content

class ACCController:
    """
    ACCのメインコントローラー。
    メモリ更新サイクルを制御する。
    """
    def __init__(self):
        # Load Context Files
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
        file_path = self.settings_dir / filename
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return ""

    def prepare_turn(self, user_input: str) -> Dict[str, Any]:
        """
        ターンの準備フェーズ (Recall, Qualify, Compress)。
        返り値として、新しいCCSと取得したアーティファクトを含む辞書を返す。
        """
        # 1. Recall (Step 2)
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
        
        return {
            "text": user_input, 
            "ccs": new_ccs,
            "qualified_artifacts": qualified_artifacts
        }

    async def stream_action(self, user_input: str) -> AsyncIterator[str]:
        """
        アクションフェーズ (Step 5) の非同期ストリーミング実行。
        """
        async for chunk in self.agent.generate_response_stream(user_input, self.current_ccs):
            yield chunk

    def finalize_turn(self, user_input: str, response_text: str):
        """
        ターンの完了処理。
        日記の更新、記憶の抽出、ベクトルDBへの保存など、重い処理をここで行う。
        """
        # --- Memory Updates (OpenClaw Style) ---
        
        # 1. Daily Log (Journal Update)
        current_journal = self.memory_manager.read_daily_journal()
        new_journal = self.memory_processor.update_daily_journal(current_journal, user_input, response_text)
        self.memory_manager.save_daily_journal(new_journal)
        
        # 2. Memory Flush (Extract Facts)
        facts = self.memory_processor.extract_memories(user_input, response_text, self.current_ccs)
        if facts:
            self.memory_manager.append_to_long_term_memory(facts)
            # Also add to vector store for retrieval
            for fact in facts:
                self.store.add_artifact(fact, metadata={"type": "semantic_memory", "source": "memory_flush"})

        # (Legacy) Episodic Trace for Artifact Store
        # 今回のCCSのコピーを保存
        self.store.add_artifact(
            content=f"User: {user_input}\nAssistant: {response_text}\nGist: {self.current_ccs.semantic_gist}",
            metadata={"type": "episodic_memory"}
        )

    def process_turn(self, user_input: str) -> Dict[str, Any]:
        """
        (Legacy/Sync) 全工程を同期的に行うメソッド。
        """
        # 1-4. Prepared
        prep_result = self.prepare_turn(user_input)
        
        # 5. Action
        response_text = self.agent.generate_response(user_input, self.current_ccs)
        
        # Finalize
        self.finalize_turn(user_input, response_text)
        
        return {
            "response": response_text,
            "ccs": self.current_ccs.model_dump()
        }
