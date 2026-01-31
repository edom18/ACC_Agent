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

    def get_daily_log_path(self, date: Optional[datetime.date] = None) -> Path:
        """指定された日付（デフォルトは今日）のDaily Logのパスを取得する"""
        if date is None:
            date = datetime.date.today()
        return self.memory_dir / f"{date.strftime('%Y-%m-%d')}.md"

    def read_recent_daily_logs(self, days: int = 2) -> str:
        """過去N日分のDaily Logを読み込む"""
        today = datetime.date.today()
        logs = []
        
        # 今日から過去N日前まで遡る（新しい順に表示するか、古い順にするか。文脈的には古い→新しいが自然）
        # days=2なら、昨日と今日。
        for i in range(days - 1, -1, -1):
            date = today - datetime.timedelta(days=i)
            log_path = self.get_daily_log_path(date)
            if log_path.exists():
                content = log_path.read_text(encoding="utf-8")
                logs.append(f"# {date.strftime('%Y-%m-%d')}\n\n{content}")
        
        return "\n\n".join(logs)

    def append_daily_log(self, content: str):
        """今日のDaily Logに追記する"""
        log_path = self.get_daily_log_path()
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        # ファイルが存在しない場合はヘッダーを作成
        if not log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"# {today_str}\n\n")

        # 追記（改行を挟む）
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{content}\n")

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
