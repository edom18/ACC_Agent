import os
from pathlib import Path

def test_context_split():
    user_name = "edom18"
    settings_dir = Path(f"agent-settings/{user_name}")
    common_settings_dir = Path("agent-settings/common")
    
    # Mock loader logic
    def _load_context_file(filename, is_common=False):
        base_dir = common_settings_dir if is_common else settings_dir
        file_path = base_dir / filename
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None

    print(f"--- Verification for user: {user_name} ---")
    
    identity = _load_context_file("IDENTITY.md")
    soul = _load_context_file("SOUL.md")
    user = _load_context_file("USER.md")
    agents = _load_context_file("AGENTS.md", is_common=True)
    
    print(f"\nResults:")
    print(f"IDENTITY.md: {'FOUND' if identity else 'NOT FOUND'} ({len(identity) if identity else 0} chars)")
    print(f"SOUL.md:     {'FOUND' if soul else 'NOT FOUND'} ({len(soul) if soul else 0} chars)")
    print(f"USER.md:     {'FOUND' if user else 'NOT FOUND'} ({len(user) if user else 0} chars)")
    print(f"AGENTS.md:   {'FOUND' if agents else 'NOT FOUND'} ({len(agents) if agents else 0} chars)")
    
    if all([identity, soul, user, agents]):
        print("\nSUCCESS: All four context files found in correct locations.")
    else:
        print("\nFAILURE: Some files were not found.")

if __name__ == "__main__":
    test_context_split()
