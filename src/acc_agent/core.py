import os
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, AsyncIterator, List
from collections import deque
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from .schemas import CompressedCognitiveState
from .memory import ArtifactStore
from .memory_manager import MemoryManager
from .introspection import IntrospectionAgent
from .llm_factory import get_llm_model

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

    def __init__(self, agents_context: str = "", model_name: Optional[str] = None):
        self.llm = get_llm_model(model_name=model_name, temperature=0.0)
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
            ("human", "Update the Cognitive State based on the input: {current_input}")
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
    def __init__(self, store: ArtifactStore, identity_context: str = "", soul_context: str = "", user_context: str = "", agents_context: str = "", model_name: Optional[str] = None):
        self.raw_llm = get_llm_model(model_name=model_name, temperature=0.7)
        self.identity_context = identity_context
        self.soul_context = soul_context
        self.user_context = user_context
        self.agents_context = agents_context
        self.store = store

        # Define and Bind Tools
        @tool
        def search_memory(query: str) -> str:
            """
            Search the agent's long-term memory and daily notes for information.
            Use this tool when the conversation context is missing information or when you need to recall past events.
            """
            if os.getenv("ACC_DEBUG", "false").lower() == "true":
                print(f"\n[ACC] 🔍 Searching Memory with query: '{query}'")

            results = self.store.recall(query, n_results=3)
            # recall returns list of strings, join them
            return "\n---\n".join(results) if results else "No relevant information found."

        self.tools = [search_memory]
        self.llm = self.raw_llm.bind_tools(self.tools)

    def generate_response(self, current_input: str, ccs: CompressedCognitiveState, recent_memory: str = "", history: List[BaseMessage] = []) -> str:
        # Construct System Prompt
        system_prompt = """あなたはAIアシスタントです。

# あなたのアイデンティティ (Identity)
{identity_context}

# あなたの内面・指針 (Soul)
{soul_context}

# 厳格に従うべきルール (Agents Protocols)
{agents_context}

# ユーザプロフィール (User Profile)
{user_context}

# 直近の記憶 (Recent Memory)
{recent_memory}

--

以下の「圧縮された認知状態 (CCS)」と「直近の会話履歴」をコンテキストとして持ち、ユーザーに応答してください。
もし情報が不足している場合は、`search_memory` ツールを使用して過去の記憶を検索してください。

# 現在の認知状態 (Current Cognitive State)
{ccs_json}

この状態に基づき、ユーザーの入力に対して適切に応答・アクションを行ってください。
制約事項 (Constraints) は必ず守ってください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{current_input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True), # For older LC versions or manual tool loop
        ])

        # Prepare initial input vars
        input_vars = {
            "ccs_json": ccs.model_dump_json(indent=2),
            "current_input": current_input,
            "identity_context": self.identity_context,
            "soul_context": self.soul_context,
            "user_context": self.user_context,
            "agents_context": self.agents_context,
            "recent_memory": recent_memory,
            "chat_history": history
        }

        # Manual Tool Execution Loop (ReAct-like)
        messages = prompt.format_messages(**input_vars)
        
        # 1. First LLM Call
        ai_msg = self.llm.invoke(messages)
        
        _log_llm_interaction("STEP 5: Action (Initial)", messages, ai_msg)

        # Loop for tool calls
        tool_iterations = 0
        while ai_msg.tool_calls and tool_iterations < 3:
            messages.append(ai_msg)
            
            for tool_call in ai_msg.tool_calls:
                selected_tool = {"search_memory": self.tools[0]}[tool_call["name"].lower()]
                tool_output = selected_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
                
                # Log tool output
                if os.getenv("ACC_DEBUG", "false").lower() == "true":
                    print(f"TOOL OUTPUT ({tool_call['name']}): {tool_output}")

            # 2. Subsequent LLM Call
            tool_iterations += 1
            ai_msg = self.llm.invoke(messages)
            _log_llm_interaction(f"STEP 5: Action (After Tool {tool_iterations})", messages, ai_msg)

        return ai_msg.content

    async def generate_response_stream(self, current_input: str, ccs: CompressedCognitiveState, recent_memory: str = "", history: List[BaseMessage] = []) -> AsyncIterator[str]:
        """
        ストリーミングレスポンスを生成する非同期ジェネレータ。
        ツール呼び出しのループ処理を含む。
        """
        system_prompt = """あなたはAIアシスタントです。

# あなたのアイデンティティ (Identity)
{identity_context}

# あなたの内面・指針 (Soul)
{soul_context}

# 厳格に従うべきルール (Agents Protocols)
{agents_context}

# ユーザプロフィール (User Profile)
{user_context}

# 直近の記憶 (Recent Memory)
{recent_memory}

--

以下の「圧縮された認知状態 (CCS)」と「直近の会話履歴」をコンテキストとして持ち、ユーザーに応答してください。
もし情報が不足している場合は、`search_memory` ツールを使用して過去の記憶を検索してください。

# 現在の認知状態 (Current Cognitive State)
{ccs_json}

この状態に基づき、ユーザーの入力に対して適切に応答・アクションを行ってください。
制約事項 (Constraints) は必ず守ってください。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{current_input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True), # For older LC versions or manual tool loop
        ])

        input_vars = {
            "ccs_json": ccs.model_dump_json(indent=2),
            "current_input": current_input,
            "identity_context": self.identity_context,
            "soul_context": self.soul_context,
            "user_context": self.user_context,
            "agents_context": self.agents_context,
            "recent_memory": recent_memory,
            "chat_history": history
        }

        # Manual Streaming Tool Execution Loop
        messages = prompt.format_messages(**input_vars)
        
        tool_iterations = 0
        while tool_iterations < 3:
            # 1. Stream First/Next LLM Call
            # We need to capture the full message to check for tool calls, 
            # while yielding content chunks to the user.
            
            ai_msg_content = ""
            tool_calls = []
            
            # Note: For streaming tool calls, we should ideally aggregate chunks.
            # However, simpler approach: stream, then if tool_calls attr exists on the final aggregated object (not easy with simple loop)
            # We will use 'astream' to yield chunks, and we also need to reconstruct the AIMessage.
            # Using 'astream_events' or similar is better, but here we can iterate and check chunks.
            
            current_tool_call = None
            
            _log_llm_interaction(f"STEP 5: Action Stream (Iter {tool_iterations})", messages, "(Streaming...)")

            ai_message_chunk = None
            
            async for chunk in self.llm.astream(messages):
                if not ai_message_chunk:
                    ai_message_chunk = chunk
                else:
                    ai_message_chunk += chunk
                
                if chunk.content:
                    if isinstance(chunk.content, list):
                        # Handle content being a list (e.g. multi-modal or specific provider behaviors)
                        # Normally it's a list of strings or dicts. If strings, join them.
                        content_str = ""
                        for item in chunk.content:
                            if isinstance(item, str):
                                content_str += item
                            elif isinstance(item, dict) and "text" in item:
                                content_str += item["text"]
                        
                        yield content_str
                    else:
                        yield chunk.content
                    
            # After streaming finishes for this turn, check if there were tool calls
            if ai_message_chunk and ai_message_chunk.tool_calls:
                # Tool call detected!
                messages.append(ai_message_chunk)
                
                # Notify user (optional, can look like a thought)
                yield "\n(Searching memory...)\n" 

                for tool_call in ai_message_chunk.tool_calls:
                    selected_tool = {"search_memory": self.tools[0]}[tool_call["name"].lower()]
                    tool_output = selected_tool.invoke(tool_call["args"])
                    
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
                    
                    if os.getenv("ACC_DEBUG", "false").lower() == "true":
                        print(f"TOOL OUTPUT ({tool_call['name']}): {tool_output}")

                tool_iterations += 1
                # Continue loop -> Re-invoke LLM with tool outputs
            else:
                # No tool calls, this was the final answer.
                break

class ACCController:
    """
    ACCのメインコントローラー。
    メモリ更新サイクルを制御する。
    """
    def __init__(self):
        # Load Context Files
        self.user_name = os.getenv("ACC_USER_NAME", "edom18")
        self.settings_dir = Path(f"agent-settings/{self.user_name}")
        self.common_settings_dir = Path("agent-settings/common")
        
        self.identity_context = self._load_context_file("IDENTITY.md")
        self.soul_context = self._load_context_file("SOUL.md")
        self.user_context = self._load_context_file("USER.md")
        self.agents_context = self._load_context_file("AGENTS.md", is_common=True)

        # Initialize Memory Components
        self.memory_manager = MemoryManager(user_name=self.user_name)
        self.introspection = IntrospectionAgent(user_name=self.user_name)

        self.ccm = CognitiveCompressorModel(agents_context=self.agents_context)
        self.store = ArtifactStore()
        self.agent = AgentEngine(
            store=self.store,
            identity_context=self.identity_context,
            soul_context=self.soul_context,
            user_context=self.user_context,
            agents_context=self.agents_context
        )
        self.history: deque = deque(maxlen=15)
        self.current_ccs: Optional[CompressedCognitiveState] = None
        self.current_recent_memory: str = ""

    def _load_context_file(self, filename: str, is_common: bool = False) -> str:
        base_dir = self.common_settings_dir if is_common else self.settings_dir
        file_path = base_dir / filename
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
        
        # Load Recent Memory for Action
        self.current_recent_memory = self.memory_manager.read_recent_daily_logs()
        
        return {
            "text": user_input, 
            "ccs": new_ccs,
            "qualified_artifacts": qualified_artifacts,
            "recent_memory": self.current_recent_memory,
            "history": list(self.history)
        }

    async def stream_action(self, user_input: str) -> AsyncIterator[str]:
        """
        アクションフェーズ (Step 5) の非同期ストリーミング実行。
        """
        async for chunk in self.agent.generate_response_stream(user_input, self.current_ccs, recent_memory=self.current_recent_memory, history=list(self.history)):
            yield chunk

    def finalize_turn(self, user_input: str, response_text: str):
        """
        ターンの完了処理。
        日記の更新、記憶の抽出、ベクトルDBへの保存など、重い処理をここで行う。
        """
        # --- Memory Updates (OpenClaw Style) ---
        
        # 1. Introspection Cycle (Journal, Facts, Context Updates)
        introspection_results = self.introspection.run_introspection_cycle(user_input, response_text, self.current_ccs)
        
        # Log Journal
        if introspection_results["journal_entry"]:
            self.memory_manager.append_daily_log(introspection_results["journal_entry"])
            
        # Log Facts
        facts = introspection_results["facts"]
        if facts:
            self.memory_manager.append_to_long_term_memory(facts)
            for fact in facts:
                self.store.add_artifact(fact, metadata={"type": "semantic_memory", "source": "memory_flush"})
        
        # Notify if Context Updated
        if introspection_results["updated_files"]:
            print(f"*** Context Updated: {introspection_results['updated_files']} ***")
            # Reload context for next turn
            if "IDENTITY.md" in introspection_results["updated_files"]:
                self.identity_context = self._load_context_file("IDENTITY.md")
                self.agent.identity_context = self.identity_context
            if "SOUL.md" in introspection_results["updated_files"]:
                self.soul_context = self._load_context_file("SOUL.md")
                self.agent.soul_context = self.soul_context
            if "USER.md" in introspection_results["updated_files"]:
                self.user_context = self._load_context_file("USER.md")
                self.agent.user_context = self.user_context
            if "AGENTS.md" in introspection_results["updated_files"]:
                self.agents_context = self._load_context_file("AGENTS.md", is_common=True)
                self.agent.agents_context = self.agents_context
                self.ccm.agents_context = self.agents_context

        # (Legacy) Episodic Trace for Artifact Store
        # 今回のCCSのコピーを保存
        self.store.add_artifact(
            content=f"User: {user_input}\nAssistant: {response_text}\nGist: {self.current_ccs.semantic_gist}",
            metadata={"type": "episodic_memory"}
        )

        # Update Sliding Window History
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response_text))