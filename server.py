import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict

from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Import from our package
from acc_agent.core import ACCController

app = FastAPI(title="ACC Agent Server")

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session storage for simplicity
# session_id -> ACCController instance
sessions: Dict[str, ACCController] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str
    state: dict

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    
    if session_id not in sessions:
        sessions[session_id] = ACCController()
    
    controller = sessions[session_id]
    
    try:
        result = controller.process_turn(request.message)
        return ChatResponse(
            reply=result["response"],
            state=result["ccs"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
