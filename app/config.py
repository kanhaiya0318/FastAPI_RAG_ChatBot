import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    GEMINI_API_KEY : str
    GEMINI_MODEL : str = "gemini-3.1-flash-lite"

    WEATHER_API_KEY : str

    RETRY_LLM_CALL : int = 4
    TRIGGER : int = 8
    KEEP_LAST : int = 4

    #web search
    SEARCH_DEPTH : Literal["advanced", "ultra-fast", "fast", "basic"] = "ultra-fast"
    SEARCH_MAX_RESULTS: int = 5 
    TAVILY_API_KEY : str

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"
    )


settings = Settings()