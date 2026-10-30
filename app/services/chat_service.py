from app.services.llm_service import LLMService
from app.services.memory_service import compress
from app.config import settings
from app.utils.timing import timer, logger
from app.services.tools import SearchDocs


async def run_tool_calls(parts: list, rag) -> list:
    response_parts = []
    for part in parts:
        call = part.get("functionCall")
        if not call:
            continue
        name = call.get("name", "")
        args = call.get("args", {})

        if name == "SearchDocs":
            with timer("SearchDocs : "):
                result = await SearchDocs(args.get("query", ""), rag)
        else:
            result = {"error": "Unknown function name"}

        fr = {"name": name, "response": result}
        if call.get("id"):
            fr["id"] = call["id"]
        response_parts.append({"functionResponse": fr})
    return response_parts


async def reply(db: dict, prompt: str, llm_service: LLMService, rag):
    db["history"].append({"role": "user", "parts": [{"text": prompt}]})

    with timer("Compress : "):
        await compress(db, llm_service)

    for i in range(settings.RETRY_LLM_CALL):
        text_buf, other_parts = "", []
        try:
            async for chunk in llm_service.generate_content_stream(
                db["history"], db["summary"]
            ):
                if "candidates" not in chunk:
                    continue
                parts = chunk["candidates"][0].get("content", {}).get("parts", [])

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
        db["history"].append({"role": "model", "parts": parts})

        if not any("functionCall" in p for p in parts):
            return

        response_parts = await run_tool_calls(parts, rag)
        db["history"].append({"role": "user", "parts": response_parts})

    yield "\n[error: the model used too many tool turns without a final answer]"