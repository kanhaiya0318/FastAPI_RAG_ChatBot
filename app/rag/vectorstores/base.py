from __future__ import annotations

import asyncio
from typing import Iterable, Any
from abc import ABC, abstractmethod

from app.rag.schema import Document
from app.rag.embeddings.base import BaseEmbeddings


__all__ = ["BaseVectorStore"]


class BaseVectorStore(ABC):

    def __init__(self, embedding :BaseEmbeddings)->None:
        self._embedding = embedding 

    @property
    def embeddings(self)->BaseEmbeddings:
        return self._embedding

    @abstractmethod
    def add_documents(
        self, documents: Iterable[Document],*, ids:list[str] | None = None
    )->list[str]:
        """Embed and store documents (adding again with the same id overwrites).

        Args:
            documents: Documents to store.
            ids: One id per document. Random ids are generated if omitted.

        Returns:
            The ids of the stored documents.
        """


    @abstractmethod
    def similarity_search_by_vector(
        self, embedding: list[float], k:int = 4, filter:dict[str,Any] | None = None
    )-> list[tuple[Document, float]]:
        """Find the documents closest to a vector.

        Args:
            embedding: The query vector.
            k: Maximum number of results.
            filter: Metadata filter (store-specific syntax).

        Returns:
            ``(document, similarity)`` pairs, most similar first.
        """

    @abstractmethod
    def delete(
        self, ids:list[str] | None = None, filter:dict[str,Any]| None = None
    )->None:
        """Delete documents by id and/or metadata filter.

        Raises:
            ValueError: If neither ``ids`` nor ``filter`` is given.
        """

    @abstractmethod
    def get(
        self,
        ids:list[str]|None = None, 
        filter:dict[str, Any]|None = None,
        limit:int |None = None,
        offset:int|None = None,
    )->list[Document]:
        """Fetch stored documents without searching."""


    @abstractmethod
    def count(self)-> int:
        """Return the number of stored documents."""


    # ---------- provided ----------
    
    # ---------- similarity ----------

    def similarity_search_with_score(
        self, query:str, k:int=4, filter:dict[str, Any]|None=None
    )->list[tuple[Document, float]]:

        vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(vector,k=k,filter=filter)


    def similarity_search(
        self, query:str, k:int=4, filter:dict[str, Any]|None=None
    )->list[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, filter)]


    # ---------- MMR ----------

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
       
        raise NotImplementedError(f"{type(self).__name__} does not support MMR")


    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        
        """Embed a text query, then run :meth:`max_marginal_relevance_search_by_vector`."""
        vector = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            vector, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )


    # ---------- Async ----------

    async def aadd_documents(
        self, documents: Iterable[Document], *, ids:list[str]|None=None
    )->list[str]:
        return await asyncio.to_thread(self.add_documents, documents,id=ids)


    async def asimilarity_search_with_score(
        self, query:str, k:int=4, filter:dict[str, Any]|None=None
    )->list[tuple[Document, float]]:
        return await asyncio.to_thread(self.similarity_search_with_score,query, k, filter)


    async def asimilarity_search(
        self, query:str, k:int=4, filter:dict[str, Any]|None=None
    )->list[Document]:
        return await asyncio.to_thread(self.similarity_search, query, k, filter)


    async def adelete(
        self, ids:list[str]|None=None, filter:dict[str,Any]|None= None
    )->None:
        await asyncio.to_thread(self.delete, ids, filter)


    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Async version of :meth:`max_marginal_relevance_search`."""
        return await asyncio.to_thread(
            self.max_marginal_relevance_search, query, k, fetch_k, lambda_mult, filter
        )