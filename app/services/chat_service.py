from fastapi import HTTPException, status
from app.services.llm_service import LLMService
from app.services.memory_service import compress
from app.config import settings
from app.utils.timing import timer, logger


async def reply(db:dict, message:str, llm_service:LLMService)->str:

    db["history"].append({"role" : "user", "parts": [{"text":message}]})

    with timer("compress : "):
        await compress(db, llm_service)
    for i in range(settings.RETRY_LLM_CALL):

        with timer(f"llm_service Call {i} : "):
            json_response = await llm_service.generate_content(db["history"], db["summary"])
    
        try:
            parts = json_response["candidates"][0]["content"]["parts"]
        except(KeyError, IndexError):
            raise HTTPException(status_code = 500, detail = "Malformed response from Gemini API.")
    
        db["history"].append({"role":"model", "parts": parts})
        return "".join(p.get("text", "") for p in parts)

        # if not any("functionCall" in p for p in parts):
        #     return "".join(p.get("text", "") for p in parts)

        # response_parts = []
        # for part in parts:
        #     if "functionCall" in part:
        #         name = part["functionCall"]["name"]
        #         args = part["functionCall"].get("args",{})
        #         call_id = part["functionCall"].get("id")

        #         if name == "WebSearch":
        #             query = args.get("query")
        #             time_range = args.get("time_range", None)
        #             filter_by_published_date = args.get("filter_by_published_date", False)
        #             with timer("WebSearch Function Call : "):
        #                 tool_response = await WebSearch(query, clients.websearch, time_range, filter_by_published_date)
        #         else: 
        #             tool_response = {"error" : "Unknown function name"}

        #         response_parts.append({
        #                 "functionResponse":{
        #                     "name" : name,
        #                     "id" : call_id,
        #                     "response" : tool_response
        #                 }})
                
        # db["history"].append({"role":"tool", "parts": response_parts})

    raise HTTPException(status_code= status.HTTP_500_INTERNAL_SERVER_ERROR)




async def reply_stream(db:dict, prompt:str, llm_service: LLMService, thinking_mode:str):
    db["history"].append({"role": "user", "parts":[{"text":prompt}]})

    with timer("Compress : "):
        await compress(db, llm_service)

    for i in range(settings.RETRY_LLM_CALL):
        text_buf, other_parts = "", []
        try:
            async for chunk in llm_service.generate_content_stream(db["history"], db["summary"], thinking_mode):
                if "candidates" not in chunk:
                    continue
                parts = chunk["candidates"][0].get("content",{}).get("parts", [])

                for part in parts:
                    if "text" in part:
                        text_buf += part.get("text", "")
                        yield part["text"]
                    else:
                        other_parts.append(part)
        except Exception as e:
            logger.error(f"Gemini stream failed: {e}")
            yield "\n[error: could not get a response from the model]"
            return
        
        parts = ([{"text": text_buf}] if text_buf else []) + other_parts

        db["history"].append({"role":"model", "parts": parts})

        if not any("functionCall" in p for p in parts):
            return

        # response_parts = []
        # for part in parts:
        #     if "functionCall" in part:
        #         name = part["functionCall"].get("name", "")
        #         args = part["functionCall"].get("args", {})
        #         call_id = part["functionCall"].get("id")

        #         if name == "WebSearch":
        #             query = args.get("query")
        #             time_range = args.get("time_range", None)
        #             filter_by_published_date = args.get("filter_by_published_date", False)
        #             with timer("WebSearch Function Call : "):
        #                 tool_response = await WebSearch(query, clients.websearch, time_range, filter_by_published_date)

        #         else: 
        #             tool_response = {"error": "Unknow function name"}
        #         response_parts.append({
        #             "functionResponse": {
        #                 "name" : name,
        #                 "id" : call_id,
        #                 "response" : tool_response
        #             }
        #         })

        # db["history"].append({"role": "tool", "parts":response_parts})

    yield "\n[error: the model used too many tool turns without a final answer]"

