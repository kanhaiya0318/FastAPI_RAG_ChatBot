"""RAG pipeline: PDF in, relevant chunks out."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from app.rag.schema import Document
from app.rag.loaders.pymupdf_loader import PyMuPDFLoader
from app.rag.preprocessing import clean_text
from app.rag.splitters.base import BaseSplitter
from app.rag.vectorstores.base import BaseVectorStore
from app.rag.retrievers.base import BaseRetriever
from app.rag.retrievers.vector import VectorStoreRetriever



__all__ = [
    "RAGPipeline", 
    "IngestResult", 
    "DocumentInfo", 
    "file_digest", 
    "DEFAULT_TEXT_FILTER"
]

logger = logging.getLogger(__name__)

DEFAULT_TEXT_FILTER = partial(clean_text, english_only=True)


def file_digest(path:str|Path, *, length:int = 16)->str:
    """Hash a file's content (SHA-256, shortened) to use as its document id.

    Args:
        path: File to hash.
        length: Number of hex characters to keep.

    Returns:
        The same id for the same bytes, whatever the file is called.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):    #reads a 2^20bytes(1MB) chunk of bytes
            digest.update(block)
    return digest.hexdigest()[:length]


def _chunk_id(doc_id:str, index:int)->str:
    return f"{doc_id}:{index}"


@dataclass(frozen=True)
class IngestResult:
    doc_id:str
    source:str
    total_pages:int
    chunks:int
    skipped:bool  #true if this exact file was already indexed


@dataclass(frozen=True)
class DocumentInfo:
    doc_id:str
    source:str
    total_pages:int
    chunks:int


class RAGPipeline:

    def __init__(
        self,
        *,
        splitter:BaseSplitter,
        vectorstore:BaseVectorStore,
        retriever:BaseRetriever | None=None,
        text_filter:Callable[[str], str] | None = DEFAULT_TEXT_FILTER
    )->None:
        self._splitter = splitter
        self._store = vectorstore
        self._retriever = retriever or VectorStoreRetriever(vectorstore)
        self._text_filter = text_filter
        

    @property
    def vectorstore(self)->BaseVectorStore:
        return self._store


    @property
    def retriever(self)->BaseRetriever:
        return self._retriever


    #-----------------ingest-------------

    def ingest(
        self,
        path:str | Path,
        *,
        source_name:str | None=None,
        force:bool = False,
    )->IngestResult:
        started = time.perf_counter()
        path = Path(path)
        doc_id = file_digest(path)
        source = source_name or path.name

        existing = self._find_complete(doc_id)
        if existing is not None and not force:
            logger.info("Skipping %s (%s): already indexed", source, doc_id)
            return IngestResult(
                doc_id=doc_id,
                source=str(existing.metadata.get("source", source)),
                total_pages=int(existing.metadata.get("total_pages", 0)),
                chunks=int(existing.metadata["chunk_total"]),
                skipped=True
            )
        old_total = int(existing.metadata["chunk_total"]) if existing else 0

        loader = PyMuPDFLoader(
            path,
            text_filter=self._text_filter,
            extra_metadata={"doc_id":doc_id, "source":source}
        )
        pages = loader.load()
        if not pages:
            raise ValueError(f"No extractable text found in {source} (scanned PDF?)")

        chunks = self._splitter.split_documents(pages)
        if not chunks:
            raise ValueError(f"No chunks produced for {source}")
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)

        self._store.add_documents(
            chunks, ids=[_chunk_id(doc_id,i) for i in range(len(chunks))]
        )
        if old_total > len(chunks):
            self._store.delete(
                ids=[_chunk_id(doc_id,i) for i in range(len(chunks), old_total)]
            )
        logger.info(
            "Ingested %s (%s): %d pages with text -> %d chunks in %.1fs",
            source, doc_id, len(pages), len(chunks), time.perf_counter() - started,
        )
        return IngestResult(
            doc_id=doc_id,
            source=source,
            total_pages=int(pages[0].metadata["total_pages"]),
            chunks=len(chunks),
            skipped=False,
        )


    async def aingest(
        self, path: str | Path, *, source_name:str | None=None, force:bool = False
    )->IngestResult:
        return await asyncio.to_thread(
            self.ingest, path, source_name=source_name, force=force
        )
    

    def _find_complete(self, doc_id:str)->Document|None:
        """Return the document's first chunk if ALL its chunks are stored.

        Chunks are written in order, so if the last one is there the earlier
        ones are too. A half-finished ingest is therefore not "already done".
        """
        first = self._store.get(ids=[_chunk_id(doc_id, 0)])
        if not first:
            return None
        total = first[0].metadata.get("chunk_total")
        if not isinstance(total, int) or total < 1:
            return None
        if not self._store.get(ids=[_chunk_id(doc_id, total-1)]):
            return None
        return first[0]
        


    #-----------------retrieve-------------

    def retrieve(
        self, query:str, *, k:int|None=None, doc_id:str|None=None
    )->list[Document]:
        """Find the chunks that best answer a question.

        Args:
            query: The user's question.
            k: Override the retriever's default number of results.
            doc_id: Search only this document.

        Returns:
            Chunks (copies) with ``metadata["score"]``, best first. May be empty.
        """
        flt = {"doc_id":doc_id} if doc_id else None
        return self._retriever.retrieve(query, k=k, filter=flt)


    async def aretrieve(
        self, query:str, *, k:int|None=None, doc_id:str|None=None
    )->list[Document]:
        flt = {"doc_id":doc_id} if doc_id else None
        return await self._retriever.aretrieve(query, k=k, filter=flt)


    #-----------------manage---------------

    def list_documents(self)->DocumentInfo:
        """List indexed documents (one entry each), sorted by name."""
        firsts = self._store.get(filter={"chunk_index":0})
        infos = [
            DocumentInfo(
                doc_id=str(d.metadata["doc_id"]),
                source=str(d.metadata.get("source","")),
                total_pages=int(d.metadata.get("total_pages", 0)),
                chunks=int(d.metadata.get("chunk_total",0)),
            ) for d in firsts if "doc_id" in d.metadata
        ]
        return sorted(infos, key=lambda i:(i.source, i.doc_id))
        
    def delete_document(self, doc_id:str)->bool:

        if not self._store.get(filter={"doc_id":doc_id}, limit=1):
            return False
        self._store.delete(filter={"doc_id":doc_id})
        return True




