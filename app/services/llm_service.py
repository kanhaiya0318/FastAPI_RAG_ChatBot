import logging
import httpx
from datetime import datetime
import json
from app.config import settings
from app.utils.timing import logger
from app.services.tools import SEARCH_DOCS_DECLARATION, SEARCH_WEB_DECLARATION
from app.rag.prompts import INSTRUCTION_PROMPT

current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
logging = logging.getLogger("uvicorn.error")

class LLMService:
    def __init__(self, client:httpx.AsyncClient):
        self.model = settings.GEMINI_MODEL
        self.api_key = settings.GEMINI_API_KEY
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.stream_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        
        self.client = client

        self.instruction_prompt = INSTRUCTION_PROMPT

        self.tools_config = [{"function_declarations": [SEARCH_DOCS_DECLARATION]}]
        # self.tools_config = [{"function_declarations": [SEARCH_WEB_DECLARATION]}]


        self.headers = {
            "content-type": "application/json",
            "x-goog-api-key" : self.api_key
        }


    async def summarize(self, prompt:str):
        
        payload = {"contents" : [{"parts": [{"text": prompt}]}]}

        try: 
            response = await self.client.post(self.url, json= payload, headers = self.headers)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise e


    async def generate_content_stream(self, prompt : list, summary : str):

        instruction_prompt = self.instruction_prompt.format(conversation_summary = summary or "(None)", datetime = current_time)
        system_instruction = {"parts": [{"text": instruction_prompt}]}

        payload = {
            "system_instruction":system_instruction,
            "contents" : prompt,
            "tools" : self.tools_config
        }

        try:
            async with self.client.stream("POST", self.stream_url, json = payload, headers = self.headers) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    logger.error(f"Gemini HTTP {response.status_code}: {body[:500]}")
                response.raise_for_status()
                async for line in response.aiter_lines(): 
                    line = line.strip()
                    if line.startswith("data: "):
                        yield json.loads(line[6:])       
        except Exception as e:
            logging.error(f"Gemini API call faild: {e}")
            raise 
