import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(os.path.abspath("src"))

from acc_agent.core import ACCController

def verify_context_loading():
    print("Testing ACCController context loading...")
    
    # Set environment variable for user
    os.environ["ACC_USER_NAME"] = "edom18"
    
    controller = ACCController()
    
    print(f"User Name: {controller.user_name}")
    print(f"Soul Context (from edom18): {len(controller.soul_context)} chars")
    print(f"User Context (from edom18): {len(controller.user_context)} chars")
    print(f"Agents Context (from common): {len(controller.agents_context)} chars")
    
    if len(controller.agents_context) > 0:
        print("SUCCESS: AGENTS.md loaded successfully.")
    else:
        print("FAILURE: AGENTS.md is empty.")

    # Verify IntrospectionAgent
    print("\nTesting IntrospectionAgent loading...")
    introspection = controller.introspection
    agents_md = introspection._read_file("AGENTS.md", is_common=True)
    print(f"Agents Context (from IntrospectionAgent): {len(agents_md)} chars")
    
    if len(agents_md) == len(controller.agents_context):
        print("SUCCESS: IntrospectionAgent read AGENTS.md correctly.")
    else:
        print("FAILURE: IntrospectionAgent read different content.")

if __name__ == "__main__":
    verify_context_loading()
