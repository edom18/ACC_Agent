import os
import datetime
from pathlib import Path
from typing import List, Optional

class MemoryManager:
    """
    OpenClawスタイルのファイルベースメモリ管理クラス。
    Daily LogsとLong-term Memory (MEMORY.md) の読み書きを担当する。
    """
    def __init__(self, user_name: str = "edom18"):
        self.user_name = user_name
        self.base_dir = Path(f"agent-settings/{self.user_name}")
        self.memory_dir = self.base_dir / "memory"
        self.memory_file = self.base_dir / "MEMORY.md"

        self._ensure_directories()

    def _ensure_directories(self):
        """必要なディレクトリとファイルが存在することを確認する"""
        if not self.memory_dir.exists():
            self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.memory_file.exists():
            self.memory_file.touch()
            self.memory_file.write_text("# Long-term Memory\n\n", encoding="utf-8")

    def get_daily_log_path(self) -> Path:
        """今日のDaily Logのパスを取得する"""
        today = datetime.date.today().strftime("%Y-%m-%d")
        return self.memory_dir / f"{today}.md"

    def append_to_daily_log(self, user_input: str, agent_response: str):
        """今日のDaily Logに対話を追加する"""
        log_path = self.get_daily_log_path()
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        entry = f"\n## [{timestamp}]\n**User**: {user_input}\n\n**Agent**: {agent_response}\n"
        
        # 追記モードで書き込み
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def append_to_long_term_memory(self, facts: List[str]):
        """MEMORY.mdに新しい事実を追加する"""
        if not facts:
            return
            
        new_content = ""
        for fact in facts:
            new_content += f"- {fact}\n"
            
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(new_content)

    def read_long_term_memory(self) -> str:
        """MEMORY.mdの内容を読み込む"""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""
