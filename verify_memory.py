import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from dotenv import load_dotenv
load_dotenv()

try:
    from acc_agent.memory_manager import MemoryManager
    from acc_agent.memory_processor import MemoryProcessor
    from acc_agent.core import ACCController
    
    print("Imports successful.")
    
    mm = MemoryManager()
    print("MemoryManager initialized.")
    logs = mm.read_recent_daily_logs()
    print(f"Read recent logs: {len(logs)} chars")
    
    mp = MemoryProcessor()
    print("MemoryProcessor initialized.")
    
    print("Verification passed.")
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
