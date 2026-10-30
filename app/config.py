from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    GEMINI_API_KEY : str
    GEMINI_MODEL : str = "gemini-3.1-flash-lite"
    GEMINI_EMBED_MODEL : str 

    RETRY_LLM_CALL : int = 5
    TRIGGER : int = 8
    KEEP_LAST : int = 4

    #web search
    SEARCH_DEPTH : Literal["advanced", "ultra-fast", "fast", "basic"] = "ultra-fast"
    SEARCH_MAX_RESULTS: int = 5 
    TAVILY_API_KEY : str

    #build rag
    CHUNK_SIZE : int = 700
    CHUNK_OVERLAP : int = 100
    OUTPUT_DIM : int = 1024
    BATCH_SIZE : int = 1

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"
    )


settings = Settings()