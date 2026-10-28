from abc import ABC, abstractmethod
from typing import Iterator
from app.rag.schema import Document

class BaseLoader(ABC):

    @abstractmethod
    def lazy_load(self)-> Iterator[Document]:
        """Yield Documents one at a time."""

    def load(self) -> list[Document]:
        return list(self.lazy_load())
