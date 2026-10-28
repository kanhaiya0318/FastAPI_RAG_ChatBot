from app.config import settings
from app.services.llm_service import LLMService
from app.utils.timing import timer, logger


def is_safe_cut(msg: dict)-> bool:
    return msg["role"] == "user"


def safe_cut(conversation_history:list, keep_last:int):

    cut = max(0, len(conversation_history)-keep_last)

    while cut > 0 and not is_safe_cut(conversation_history[cut]):
        cut -= 1
    return cut


def flatten(old: list)->str:
    lines = []

    for items in old:
        for item in items["parts"]:
            if "text" in item: 
                lines.append(f"{items["role"]} : {item["text"]}")
            if "functionCall" in item: 
                lines.append(f"{items["role"]} : Functioncall-> name: {item["functionCall"]["name"]}, args->{str(item["functionCall"]["args"])}")
            if "functionResponse" in item:
                lines.append(f"{items["role"]} : functionResponse-> name: {item["functionResponse"]["name"]}, response:{str(item["functionResponse"]["response"])}")
    return "\n".join(lines)



SUMMARY_PROMPT = """Progressively summarize the conversation, adding new lines to the existing summary.
Keep: names, preferences, decisions, tool results. Max 150 words.
    
    Existing Summary :
    {summary}

    New Lines :
    {lines}

    New Summary:"""


async def compress(db:dict, llm_service:LLMService):

    if len(db["history"]) <= settings.TRIGGER:
        return False
    cut = safe_cut(db["history"], settings.KEEP_LAST)
    if cut <= 0:
        return False
    
    summary = ""
    try:
        old = db["history"][:cut]
        summary_prompt = SUMMARY_PROMPT.format(summary=db["summary"] or "(None)", lines= flatten(old))
        with timer("summarizer : "):
            summary = await llm_service.summarize(str(summary_prompt))
    except Exception as e:
        logger.error(f"Summarization got Error: {e}")

    if len(summary) > 3:
        db["summary"] = summary
        db["history"][:] = db["history"][cut:]
    return True


    
