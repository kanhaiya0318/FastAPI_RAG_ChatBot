from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id : str = Field(..., min_length=3)
    thinking_mode : Literal["minimal", "low", "medium", "high"] = "minimal"
    message : str =  Field(..., min_length=1, max_length=200)

class ChatResponse(BaseModel):
    session_id : str
    response : str

class HistoryResponse(BaseModel):
    session_id : str
    summary : str = ""
    history : Optional[list] = None

class EndSessionResponse(BaseModel):
    status : str 
    detail : str


