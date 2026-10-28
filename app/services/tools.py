from typing import Literal
import httpx
from app.config import settings
from app.utils.timing import logger


async def WebSearch(
        query: str, 
        client: httpx.AsyncClient, 
        time_range:Literal["year", "month", "week", "day"] | None = None, 
        filter_by_published_date:bool = False
    ):
    try:
        url = "https://api.tavily.com/search"
        payload={"query": query,
                "max_results" : settings.SEARCH_MAX_RESULTS,
                "time_range": time_range,
                "include_images": False,
                "filter_by_published_date": filter_by_published_date, 
            "search_depth": settings.SEARCH_DEPTH
            }
        logger.info(f"response of Websearch: {payload}")
        header={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"}

        response = await client.post(url, json = payload, headers = header)
        response.raise_for_status()

        response_json = response.json()
        results = []
        for i in response_json["results"][:3]:
            result = {"url": i.get("url", ""),
                    "title": i.get("title",""),
                    "content":i.get("content", "")[:300],
                    "published_date": i.get("published_date", "")
            }
            results.append(result)
        logger.info(f"response of Websearch: {results}")
        return {"results":results}
    except Exception as e:
        logger.error(f"Web Search Faild {e}")
        return {"error":"Could not search the web right now."}