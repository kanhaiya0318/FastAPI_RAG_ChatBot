from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.dependancies import get_db, get_llm_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService
from app.services.chat_service import reply, reply_stream


router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)
    

@router.post("/message", response_model = ChatResponse)
async def send_message(
    payload: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
    db: dict = Depends(get_db)
):
    session_id = payload.session_id
    message = payload.message
    
    if session_id not in db:
        db[session_id] = {"summary":"", "history":[]}


    response_str = await reply(db[session_id], message, llm_service)
    return ChatResponse(session_id=session_id, response=response_str)
    



@router.post("/message_stream")
async def chat_stream(
    payload: ChatRequest,
    db:dict = Depends(get_db),
    llm_service:LLMService = Depends(get_llm_service)
):
    session_id = payload.session_id
    message = payload.message
    thinking_mode = payload.thinking_mode

    if session_id not in db:
        db[session_id] = {"summary":"", "history":[]}

    return StreamingResponse(
        reply_stream(db[session_id], message, llm_service, thinking_mode),
        media_type="text/plain",
    )
