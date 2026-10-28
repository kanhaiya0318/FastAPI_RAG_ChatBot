import logging
import httpx
from datetime import datetime
import json
from app.config import settings
from app.utils.timing import logger

current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
logging = logging.getLogger("uvicorn.error")

class LLMService:
    def __init__(self, client:httpx.AsyncClient):
        self.model = settings.GEMINI_MODEL
        self.api_key = settings.GEMINI_API_KEY
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.stream_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        
        self.client = client

        self.instruction_prompt ="""
                    "You are helpful AI Policy ChatBot."
                    "Use the available tools when necessary."
                    "After receiving tool results, provide a clear final answer to the user."

                    Current data and time use if need: {datetime}

                    Summary:{conversation_summary}
                    """

        self.tools_config = [{
            "function_declarations":[{
                    "name" : "WebSearch",
                    "description": "Search the web for current or recent information: news, sports results,prices, or facts you are unsure about.",
                    "parameters" : {
                        "type" : "OBJECT",
                        "properties" : {
                            "query":{
                                "type" : "STRING",
                                "description" : "A sort specific search queary"
                            },
                            "time_range": {
                                "type" : "STRING",
                                "enum":["day", "month", "week","year"],
                                "description": "Only return results published or updated within this window back from today. Use 'day' for today's news, 'week' for this week.Omit for timeless questions."
                            },
                            "filter_by_published_date":{
                                "type":"BOOLEAN",
                                "description":"If true, drop results without a detectable date or outside "
                                                "time_range. Use only together with time_range, when freshness matters."
                            }},
                        "required": ["query"]
                    }
                }]
        }]


        self.headers = {
            "content-type": "application/json",
            "x-goog-api-key" : self.api_key
        }


    async def generate_content(self, prompt : list, summary : str) -> dict:

        instruction_prompt = self.instruction_prompt.format(conversation_summary = summary or "(None)", datetime = current_time)
        system_instruction = {"parts": [{"text": instruction_prompt}]}

        payload = {
            "system_instruction":system_instruction,
            "contents" : prompt,
            #"tools" : self.tools_config
        }

        try:
            response = await self.client.post(self.url, json = payload, headers = self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Gemini API call faild: {e}")


    async def summarize(self, prompt:str):
        
        payload = {"contents" : [{"parts": [{"text": prompt}]}]}

        try: 
            response = await self.client.post(self.url, json= payload, headers = self.headers)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise e



    async def generate_content_stream(self, prompt : list, summary : str, thinking_mode: str):

        instruction_prompt = self.instruction_prompt.format(conversation_summary = summary or "(None)", datetime = current_time)
        system_instruction = {"parts": [{"text": instruction_prompt}]}

        payload = {
            "system_instruction":system_instruction,
            "contents" : prompt,
            #"tools" : self.tools_config,
            "generationConfig" : {"thinkingConfig":{"thinkingLevel":thinking_mode}}
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
