import copy
import logging
from abc import ABC, abstractmethod
from typing import Callable, Literal, Any, Iterable

from app.rag.schema import Document

logger = logging.getLogger(__name__)
__all__ = ["BaseSplitter"]

class BaseSplitter(ABC):

    def __init__(self,
                 chunk_size : int = 800,
                 chunk_overlap : int = 100,
                 length_function : Callable[[str], int] = len,
                 keep_separator : bool | Literal["start", "end"] = False,
                 add_start_index : bool = False,
                 strip_whitespace : bool = True
                 ) -> None:

        if chunk_size <= 0: 
            raise ValueError(f"chunk_size must be positive but you pass {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap > chunk_size:
            raise ValueError(f"chunk_overlap must be positive and smaller then chunk_size"
                             f"you are pass chunk_overlap : {chunk_overlap} and chunk_size : {chunk_size}")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace


    @abstractmethod
    def split_text(self, text:str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """


    def create_documents(self, texts : list[str], metadatas : list[dict[str, Any]] | None = None) -> list[Document]:

        metadatas_ = metadatas or [{}] * len(texts)
        Documents:list[Document] = []

        for text, base_metadata in zip(texts, metadatas_):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(base_metadata)
                if self._add_start_index:    
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                Documents.append(Document(page_content=chunk, metadata=metadata))
        return Documents


    def split_documents(self, documents: Iterable[Document])->list[Document]:

        texts : list[str] = []
        metadatas : list[dict[str, Any]] = []

        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas = metadatas)


    def _join_docs(self, docs:list[str], separator:str)->str | None:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text or None


    def _merge_splits(self, splits:Iterable[str], separator:str)->list[str]:

        separator_length = self._length_function(separator)
        docs:list[str] = []
        current_doc: list[str] = []
        total = 0

        for piece in splits:
            piece_len = self._length_function(piece)
            if total + piece_len + (separator_length if current_doc else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %d, which is longer than the "
                        "specified %d",
                        total,
                        self._chunk_size,
                    )
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)

                    # Drop pieces from the front while we hold more than the
                    # allowed overlap, or the new piece still would not fit.
                    while total > self._chunk_overlap or (
                            total + piece_len + (separator_length if current_doc else 0) 
                             > self._chunk_size 
                             and total > 0
                    ):
                            total -= self._length_function(current_doc[0]) + (
                                separator_length if len(current_doc)>1 else 0
                            )
                            current_doc = current_doc[1:]

            current_doc.append(piece)
            total += self._length_function(piece) + (separator_length if len(current_doc)>1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)

        return docs