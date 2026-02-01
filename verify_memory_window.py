import os
import sys
import asyncio
from typing import List

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from acc_agent.core import ACCController
from acc_agent.schemas import CompressedCognitiveState
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

async def main():
    load_dotenv()
    print("Initializing ACCController...")
    # Mock environment variables if needed
    os.environ["ACC_LLM_PROVIDER"] = "openai" # or gemini
    
    controller = ACCController()
    
    print("\n--- Test 1: Sliding Window History ---")
    
    # Simulate 20 turns
    print("Simulating 20 turns...")
    for i in range(1, 21):
        user_input = f"User Message {i}"
        ai_response = f"AI Response {i}"
        
        # Manually appending to history to simulate conversation flow without calling full process_turn (saves tokens/time)
        controller.history.append(HumanMessage(content=user_input))
        controller.history.append(AIMessage(content=ai_response))
        
    print(f"History Length: {len(controller.history)} (Expected 15)")
    
    if len(controller.history) == 15:
        print("✅ History length is correct.")
    else:
        print(f"❌ History length is incorrect: {len(controller.history)}")
        
    # Check content: Should be messages from 13 to 20 (approx 7.5 turns -> 15 messages)
    # Actually wait. 20 turns * 2 messages = 40 messages. Maxlen is 15.
    # So it should contain the last 15 messages.
    # The last message added was AI Response 20.
    # The deque keeps the *newest*.
    
    history_list = list(controller.history)
    print(f"Oldest message in history: {history_list[0].content}")
    print(f"Newest message in history: {history_list[-1].content}")
    
    # If 15 messages, index 0 should be...
    # Total added: 40. Keep 15. Drop 25.
    # Messages in order: U1, A1, U2, A2 ... U20, A20.
    # Dropped 0..24. Keep 25..39.
    # Message 25 (0-indexed) is...
    # Pair i (1-based) corresponds to indices 2*(i-1) and 2*(i-1)+1.
    # Message 24 was U13 (index 24). So Message 25 is A13.
    # Message 26 is U14.
    
    expected_oldest_content_substr = "13" # Or 14, depending on exact calc
    
    if "Response 20" in history_list[-1].content:
        print("✅ Newest message is correct.")
    else:
        print("❌ Newest message incorrect.")

    print("\n--- Test 2: Search Tool ---")
    
    # Add a mock artifact
    secret_code = "The secret launch code is ALPHA-BETA-GAMMA."
    controller.store.add_artifact(secret_code, metadata={"source": "test"})
    print(f"Added artifact: {secret_code}")
    
    # Verify tool exists
    search_tool = None
    for tool in controller.agent.tools:
        if tool.name == "search_memory":
            search_tool = tool
            break
            
    if search_tool:
        print("✅ Search tool found in agent.")
        
        # Invoke tool manually
        query = "What is the secret launch code?"
        print(f"Invoking tool with query: {query}")
        result = search_tool.invoke({"query": query})
        
        print(f"Tool Result: {result}")
        
        if "ALPHA-BETA-GAMMA" in result:
             print("✅ Tool successfully retrieved artifact.")
        else:
             print("❌ Tool failed to retrieve artifact.")
             
    else:
        print("❌ Search tool NOT found.")

if __name__ == "__main__":
    asyncio.run(main())
