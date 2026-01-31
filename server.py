import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Iterator

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

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

async def stream_generator(controller: ACCController, user_input: str, background_tasks: BackgroundTasks) -> Iterator[str]:
    """
    ストリーミングレスポンスを生成するジェネレータ。
    """
    # 1-4. Preparation (Synchronous wait)
    # ここでRecall, Qualify, Compressが走る
    try:
        prep_result = controller.prepare_turn(user_input)
    except Exception as e:
        yield f"Error during preparation: {str(e)}"
        return

    full_response = ""
    
    # 5. Action (Streaming)
    try:
        async for chunk in controller.stream_action(user_input):
            full_response += chunk
            yield chunk
    except Exception as e:
        yield f"[Error generating response: {str(e)}]"
        return

    # 完了後にメモリ保存タスクを予約
    # BackgroundTasksはResponse返却後に実行される
    background_tasks.add_task(controller.finalize_turn, user_input, full_response)

    # 最後にCCSの状態をJSONデータとして送る (SSEのデータイベントとして送るのが一般的だが、
    # ここではシンプルにテキストストリームの後に区切り文字を入れて送る、あるいはクライアントが別途取得する設計も可。
    # 今回はシンプルにテキストのみを返すストリームとし、Stateの即時反映は諦めるか、
    # クライアント側で別途ポーリングしてもらう想定とする。
    # 要件は「レスポンスの遅さを最小にする」なので、Textだけ返ればOKとする。)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id
    
    if session_id not in sessions:
        sessions[session_id] = ACCController()
    
    controller = sessions[session_id]
    
    return StreamingResponse(
        stream_generator(controller, request.message, background_tasks),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"}
    )

@app.get("/state/{session_id}")
async def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    controller = sessions[session_id]
    if controller.current_ccs:
        return controller.current_ccs.model_dump()
    else:
        return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
