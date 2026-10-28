from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod

__all__ = ["EmbeddingError", "BaseEmbeddings"]

class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails or returns bad data."""


class BaseEmbeddings(ABC):

    @abstractmethod
    def embed_documents(self, texts:list[str])->list[list[float]]:
        """Embed a list of documents (the chunks you store).

        Args:
            texts: Texts to embed.

        Returns:
            One vector per text, in the same order.
        """

    @abstractmethod
    def embed_query(self, text:str)->list[float]:
        """Embed a search query.

        Args:
            text: The query text.

        Returns:
            The query vector.
        """

    async def aembed_documents(self, texts:list[str])->list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text:str)->list[float]:
        return await asyncio.to_thread(self.embed_query, text)
