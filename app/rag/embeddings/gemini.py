from __future__ import annotations

import re
import string
import logging
import time
import math
import random
from typing import Any, Callable
import httpx

from app.config import settings
from app.rag.embeddings.base import BaseEmbeddings, EmbeddingError

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 10_000
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_MAX_TOKEN_PER_BATCH = 1000
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_SPLIT = re.compile(f"([{re.escape(string.punctuation + chr(9) + chr(10) + ' ')}])")

__all__ = ["GeminiEmbeddings"]


def _estimate_tokens(text:str)->int:
    return len([p for p in _SPLIT.split(text) if p]) * 2


def _prepare_batches(texts:list[str], batch_size:int)->list[list[str]]:

    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = _estimate_tokens(text)
        if tokens > _MAX_TOKEN_PER_BATCH:
            if current:
                batches.append(current)
                current, current_tokens = [], 0
            batches.append([text])
            continue
        if current and (
            current_tokens + tokens > _MAX_TOKEN_PER_BATCH
            or len(current)==batch_size
        ):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches
            

def _12_normalize(vector:list[float])->list[float]:
    norm = math.sqrt(sum(v*v for v in vector))
    return [v / norm for v in vector] if norm else vector



class GeminiEmbeddings(BaseEmbeddings):

    def __init__(
            self,
            api_key:str,
            *,
            model:str = settings.GEMINI_EMBED_MODEL,
            output_dimensionality:int | None = 768,
            batch_size: int = _MAX_BATCH_SIZE,
            normalize:bool = True,
            timeout: float = 30.0,
            max_retries:int = 3,
            base_url:str = _BASE_URL,
            client:httpx.Client |None = None,
    )->None:

        if not api_key:
            raise ValueError("API key must not be Empty.")
        if not 1 <= batch_size <= _MAX_BATCH_SIZE:
            raise ValueError(f"batch size must be between 1 to {_MAX_BATCH_SIZE}")
        if output_dimensionality is not None and output_dimensionality <=0:
            raise ValueError("Output dimention must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be positive")

        self.model = model
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_retries = max_retries
        self._url = f"{base_url}/models/{self.model}:batchEmbedContents"
        self._header = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout = timeout)
        self._sleep: Callable[[float], None] = time.sleep



    def embed_documents(self, texts:list[str])->list[list[float]]:

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"text[{i}] is blank")
            
        vectors: list[list[float]] = []
        for batch in _prepare_batches(texts, self.batch_size):
            vectors.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return vectors
        
    def embed_query(self, text:str)->list[float]:

        if not text or not text.strip():
            raise ValueError("query text is blank")
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]


    def close(self)-> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self)->GeminiEmbeddings:  # Now you can call by with ...as....
        return self

    def __exit__(self, *exc:object)->None:
        self.close()


    def _embed_batch(self, texts:list[str], task_type:str)->list[list[float]]:
        requests: list[dict[str, Any]] = []

        for text in texts:
            request: dict[str, Any] = {
                "model": f"models/{self.model}",
                "content":{"parts":[{"text":text}]},
                "taskType": task_type,
            }
            if self.output_dimensionality:
                request["outputDimensionality"] = self.output_dimensionality
            requests.append(request)

        data = self._post({"requests":requests})
        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, got "
                f"{len(embeddings) if isinstance(embeddings, list) else 'none'}"
            )
        
        vectors = [[float(v) for v in e["values"]] for e in embeddings]
        return [_12_normalize(v) for v in vectors] if self.normalize else vectors


    def _post(self, payload: dict[str, Any])->dict[str, Any]:
        last_error = "unknown error"

        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                response = self._client.post(
                    self._url, headers=self._header, json=payload
                )
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            else:
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in _RETRY_STATUS:
                    raise EmbeddingError(f"Embedding request failed ({last_error})")

                header = response.headers.get("retry-after")
                if header and header.replace(".", "", 1).isdigit():
                    retry_after = float(header)

            if attempt < self.max_retries:
                delay = retry_after or min(2**attempt + random.random(), 30.0)
                logger.warning(
                    "Embedding request failed (%s); retry %d/%d in %.1fs",
                    last_error, attempt + 1, self.max_retries, delay,
                )
                self._sleep(delay)

        raise EmbeddingError(
            f"Embedding request failed after {self.max_retries + 1} attempts "
            f"({last_error})"
        )

