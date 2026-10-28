from __future__ import annotations

import re
import unicodedata


__all__ = [
    "normalize_unicode",
    "keep_english",
    "normalize_whitespace",
    "fix_hyphenation",
    "rejoin_lines",
    "clean_text",
]

# characters that add nothing: soft hyphen, zero-width space, BOM, form feed
_INVISIBLE = re.compile(r"[\u00ad\u200b\ufeff\x0c]")
# a line that starts a new list item / numbered clause: "a." "(b)" "iv." "12." "•"
_LIST_START = re.compile(
    r"^(\(?([a-zA-Z]|\d{1,3}|[ivxIVX]{1,5})[.)]|[•\-–*])(\s|$)"
)
_DEVNAGARI = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF\u200c\u200d]")
_WORD = re.compile(r"[A-Za-z]{3,}")
_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")


def normalize_unicode(text:str)->str:
    text = unicodedata.normalize("NFKC", text)
    return _INVISIBLE.sub("", text)

def keep_english(text:str)->str:
    lines: list[str] = []
    for line in text.splitlines():
        if not _DEVNAGARI.search(line):
            lines.append(line)
            continue
        tokens:list[str] = []
        for token in line.split():
            if _DEVNAGARI.search(token):
                token = _DEVNAGARI.sub("", token)
                if not _HAS_ALNUM.search(token):
                    continue
            tokens.append(token)
        cleaned = " ".join(tokens).strip(" /|")
        if cleaned :
            lines.append(cleaned)
    return "\n".join(lines)
        

def normalize_whitespace(text:str)->str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]     # split on \n
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def fix_hyphenation(text:str)->str:
    return re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

def rejoin_lines(text:str)->str:
    paragraphs: list[str]= []
    for para in text.split("\n\n"):
        items: list[str] = []
        buf = ""
        for line in para.split("\n"):
            if not line:
                continue
            elif buf and _LIST_START.match(line):
                items.append(buf)
                buf = line
            elif buf:
                buf = f"{buf} {line}"
            else:
                buf = line
        if buf:
            items.append(buf)
        if items:
            paragraphs.append("\n".join(items))
    return "\n\n".join(paragraphs)

def clean_text(text:str, *,english_only:bool = False)->str:
    text = normalize_unicode(text)
    if english_only:
        text = keep_english(text)
    text = normalize_whitespace(text)
    text = fix_hyphenation(text)
    text = rejoin_lines(text)
    return text

    