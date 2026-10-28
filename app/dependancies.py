from fastapi import Request
from app.services.llm_service import LLMService

IN_MEMORY_DB = {}

def get_db():
    return IN_MEMORY_DB

def get_llm_service(request: Request)->LLMService:
    return LLMService(client=request.app.state.gemini)

