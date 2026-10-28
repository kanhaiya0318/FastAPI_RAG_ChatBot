from __future__ import annotations

from pathlib import Path
import uuid
import warnings
from typing import Iterable, Any

import chromadb

from app.rag.vectorstores.utils import maximal_marginal_relevance
from app.rag.vectorstores.base import BaseVectorStore
from app.rag.schema import Document
from app.rag.embeddings.base import BaseEmbeddings, EmbeddingError

__all__ = ["ChromaVectorStore"]

class ChromaVectorStore(BaseVectorStore):

    def __init__(
            self,
            embedding:BaseEmbeddings,
            *,
            collection_name:str = "documents",
            persist_directory:str| Path = "data/chroma",
            client: chromadb.ClientAPI | None= None,
    )->None:
        super().__init__(embedding)
        self._client = client or chromadb.PersistentClient(path = str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw":{"space":"cosine"}}
        )
        with warnings.catch_warnings():  # Chroma warns on reading its own config
            warnings.simplefilter("ignore", DeprecationWarning)
            space = (self._collection.configuration.get("hnsw") or {}).get("space")
        if space and space != "cosine":
            raise ValueError(
                f"Collection '{collection_name}' uses '{space}' distance, but "
                "this store needs 'cosine'. Use another collection name or "
                "delete the old collection."
            )


    def add_documents(
        self, documents: Iterable[Document],* , ids:list[str]|None=None
    )->list[str]:
        docs = list(documents)
        if not docs:
            return []
        if ids is None:
            ids = [uuid.uuid4().hex for _ in docs]
        elif len(ids) != len(docs):
            raise ValueError(f"Got {len(ids)} ids for {len(docs)} documents")
        if len(ids) != len(set(ids)):
            raise ValueError("ids must be unique")

        texts = [d.page_content for d in docs]
        vectors = self._embedding.embed_documents(texts)
        metadatas = [d.metadata or None for d in docs] # Chroma rejects {}
        if len(vectors) != len(docs):
            raise EmbeddingError(f"Expected {len(docs)} vectors, got {len(vectors)}")

        step = self._client.get_max_batch_size()
        for start in range(0, len(docs), step):
            end = start + step
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
        return ids

    # ---------- similarity ----------

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=filter or None,
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[Document, float]] = []
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            if text is None:
                continue
            similarity = max(-1.0, min(1.0, 1.0-distance))  # cosine val (0 to 2)
            out.append((Document(page_content=text, metadata=metadata or {}), similarity))
        return out


    # ---------- MMR ----------

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if fetch_k < k:
            raise ValueError(f"fetch_k ({fetch_k}) must be >= k ({k})")
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=fetch_k,
            where=filter or None,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        texts = result["documents"][0]
        if len(texts) == 0:
            return []
        chosen = maximal_marginal_relevance(
            embedding, result["embeddings"][0], k=k, lambda_mult=lambda_mult
        )
        metadatas, distances = result["metadatas"][0], result["distances"][0]
        return [
            (
                Document(page_content=texts[i], metadata=metadatas[i] or {}),
                max(-1.0, min(1.0, 1.0 - distances[i])),
            )
            for i in chosen
            if texts[i] is not None
        ]

    
    def delete(
        self, ids: list[str] | None = None, filter: dict[str, Any] | None = None
    ) -> None:
        if not ids and not filter:
            raise ValueError("Pass ids and/or filter; refusing to delete everything")
        self._collection.delete(ids=ids or None, where=filter or None)
        
    def get(
        self,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Document]:
        result = self._collection.get(
            ids=ids, where=filter or None, limit=limit, offset=offset,
            include=["documents", "metadatas"],
        )
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(result["documents"], result["metadatas"])
            if text is not None
        ]
    
        
    def count(self)->int:
        return self._collection.count()


    