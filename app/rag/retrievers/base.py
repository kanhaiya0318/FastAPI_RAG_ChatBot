from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from app.rag.schema import Document

__all__ = ["BaseRetriever"]

class BaseRetriever(ABC):

    def retrieve(
        self,
        query:str,
        *,
        k:int|None=None,
        filter:dict[str,Any] | None=None
    )->list[Document]:
        if not query or not query.strip():
            raise ValueError("query must not be blank")
        return self._get_relevant_documents(query, k=k, filter=filter)

    async def aretrieve(
        self,
        query:str,
        *,
        k:int|None=None,
        filter:dict[str,Any] |None =None,
    ):
        if not query or not query.strip():
            raise ValueError("query must not be blank")
        return await asyncio.to_thread(self.retrieve, query, k=k, filter=filter)

    @abstractmethod
    def _get_relevant_documents(
        self,
        query:str,
        *,
        k:int|None=None,
        filter: dict[str,Any]|None=None,
    )->list[Document]:
        """Do the actual retrieval (``query`` is already validated)."""