from __future__ import annotations

import re
from typing import Literal, Any
from app.rag.splitters.base import BaseSplitter

__all__ = ["RecursiveCharacterTextSplitter"]


def _split_text_with_regax(text:str, separator: str,*, keep_separator:bool | Literal["start", "end"])->list[str]:

    if not separator:
        return list(text)
    if not keep_separator:
        return [s for s in re.split(separator, text) if s]
    
    parts = re.split(f"({separator})", text)   # return always odd so no need of len(parts)%2 ==0:
    if keep_separator == "start":
        splits = [parts[0]]
        splits += [parts[i]+ parts[i+1] for i in range(1, len(parts)-1,2)]
    else: # true or # end
        splits = [parts[i]+parts[i+1] for i in range(0, len(parts)-1,2)]
        splits.append(parts[-1]) 

    return [s for s in splits if s]


class RecursiveCharacterTextSplitter(BaseSplitter):

    def __init__(self,
        separators:list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_ragax: bool = False,
        **kwargs: Any
        )->None:

        super().__init__(keep_separator = keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self._is_separator_ragax = is_separator_ragax


    def _split_text(self, text:str, separators:list[str]):
        final_chunks:list[str] = []

        # Pick the first separator that actually occurs in the text.
        separator = separators[-1]
        remaining : list[str] = []
        for i, candidate in enumerate(separators):
            pattern = candidate if self._is_separator_ragax else re.escape(candidate)
            if not candidate:
                separator = candidate
                break
            if re.search(pattern, text):
                separator = candidate
                remaining = separators[i+1:]
                break

        pattern = separator if self._is_separator_ragax else re.escape(separator)
        splits = _split_text_with_regax(
            text, pattern, keep_separator = self._keep_separator
        )

        # Merge short pieces; recurse into pieces that are too long.
        good_splits: list[str] = []
        join_with = "" if self._keep_separator else separator
        for piece in splits:
            if self._length_function(piece) <= self._chunk_size:
                good_splits.append(piece)
                continue
            if good_splits:
                final_chunks.extend(self._merge_splits(good_splits, join_with))
                good_splits = []
            if not remaining:
                final_chunks.append(piece)
            else:
                final_chunks.extend(self._split_text(piece, remaining))
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, join_with))
        return final_chunks


    def split_text(self, text:str)->list[str]:
        return self._split_text(text, self._separators)