from __future__ import annotations

from typing import Any, Literal

from app.rag.retrievers.base import BaseRetriever
from app.rag.schema import Document
from app.rag.vectorstores.base import BaseVectorStore

__all__ = ["VectorStoreRetriever"]

SearchType = Literal["similarity", "similarity_score_threshold", "mmr"]
_SEARCH_TYPES = ("similarity", "similarity_score_threshold", "mmr")

class VectorStoreRetriever(BaseRetriever):

    def __init__(
        self,
        vectorstore:BaseVectorStore,
        *,
        search_type: SearchType = "similarity",
        k: int = 4,
        score_threshold:float | None=None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
        filter: dict[str, Any] | None=None,
    )->None:

        if search_type not in _SEARCH_TYPES:
            raise ValueError(
                f"search_type must be one of {_SEARCH_TYPES}, got {search_type!r}"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if search_type == "similarity_score_threshold":
            if score_threshold is None:
                raise ValueError(
                    "score_threshold is required for 'similarity_score_threshold'"
                )
            if not -1.0 <= score_threshold <= 1.0:
                raise ValueError(
                    f"score_threshold must be between -1 and 1, got {score_threshold}"
                )
        elif score_threshold is not None:
            raise ValueError(
                "score_threshold only applies to 'similarity_score_threshold'"
            )
        if search_type == "mmr":
            fetch_k = 20 if fetch_k is None else fetch_k
            lambda_mult = 0.5 if lambda_mult is None else lambda_mult
            if fetch_k < k:
                raise ValueError(f"fetch_k ({fetch_k}) must be >= k ({k})")
            if not 0.0 <= lambda_mult <= 1.0:
                raise ValueError(f"lambda_mult must be between 0 and 1, got {lambda_mult}")
        elif fetch_k is not None or lambda_mult is not None:
            raise ValueError("fetch_k and lambda_mult only apply to 'mmr'")
        
        self._vectorstore = vectorstore
        self.search_type = search_type
        self.k = k
        self.score_threshold = score_threshold
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.filter = filter

    @property
    def vectorstore(self) -> BaseVectorStore:
        """The store this retriever searches."""
        return self._vectorstore


    def _get_relevant_documents(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        k = self.k if k is None else k
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        where = self.filter if filter is None else filter

        if self.search_type == "mmr":
            results = self._vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=max(self.fetch_k, k),  # a per-call k may exceed fetch_k
                lambda_mult=self.lambda_mult,
                filter=where,
            )
        else:
            results = self._vectorstore.similarity_search_with_score(
                query, k=k, filter=where
            )
            if self.search_type == "similarity_score_threshold":
                results = [(d, s) for d, s in results if s >= self.score_threshold]

        return [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "score": score})
            for doc, score in results
        ]