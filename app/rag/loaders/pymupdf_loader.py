from __future__ import annotations

from typing import Callable , Literal, Any
from pathlib import Path
from threading import Lock

import pymupdf

from app.rag.schema import Document
from app.rag.loaders.base import BaseLoader

__all__ = ["PyMuPDFLoader"]

class PyMuPDFLoader(BaseLoader):
    _lock = Lock()
    def __init__(
        self,
        file_path: str | Path | None = None, 
        *,
        stream : bytes | None = None,
        password : str | None = None,
        text_kwargs : dict[str, Any] | None = None,
        text_filter : Callable[[str], str] | None = None,
        extra_metadata : dict[str, Any] | None = None,
        extract_tables : Literal["markdown"] | None = None,
        skip_empty : bool = True
    )->None:

        if (file_path is None) == (stream is None):
            raise ValueError("Provide exectly one of the filepath or stream")
        if extract_tables not in ("markdown", None):
            raise ValueError("extract table must be None or markdown")

        self.file_path = Path(file_path) if file_path is not None else None
        self.stream = stream
        self.password = password
        self.text_kwargs = text_kwargs or {}
        self.text_filter = text_filter
        self.extra_metadata = extra_metadata or {}
        self.extract_tables = extract_tables
        self.skip_empty = skip_empty

    def _open(self)-> pymupdf.Document:
        if self.file_path is not None:
            if not self.file_path.exists():
                raise FileNotFoundError(f"file not found at : {self.file_path}")
            doc = pymupdf.open(self.file_path) 
        else:
            doc = pymupdf.open(stream = self.stream, filetype="pdf")  # for InMemoryDataBase
        if doc.needs_pass:
            if not self.password or not doc.authenticate(self.password):
                doc.close()
                raise ValueError("Pdf is Encripted and provided password is missing or not currect")
        return doc

    def _page_text(self, page:pymupdf.Page)-> str:
        text = page.get_text("text", **self.text_kwargs)

        if self.extract_tables == "markdown":
            tables = page.find_tables().tables     # list of tables
            if tables:
                text += "\n\n" + "\n\n".join(t.to_markdown() for t in tables)
        if self.text_filter:
            text = self.text_filter(text)
        return text.strip()

    def lazy_load(self):
        base_source = self.file_path.name if self.file_path else "stream.py"

        with PyMuPDFLoader._lock:
            doc = self._open()
            try:
                total_lenght = len(doc)
                for page in doc:
                    text = self._page_text(page)
                    if self.skip_empty and not text:
                        continue

                    yield Document(
                        page_content = text,
                        metadata = {
                            "source":base_source,
                            "page":page.number,
                            "page_number":page.number+1,
                            "total_pages": total_lenght,
                            **self.extra_metadata,
                        },
                    )
            finally:
                doc.close()
