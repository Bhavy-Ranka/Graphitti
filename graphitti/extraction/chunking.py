import re

from graphitti.config import CHUNK_MAX_WORDS

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_REFERENCE_MARKER_RE = re.compile(
    r"\b(Retrieved|ISBN|ISSN|OCLC|doi:|Archived from the original|pp?\.\s?\d)\b", re.I
)
_FOOTNOTE_MARK_RE = re.compile(r"[\^↑]")


def _looks_like_reference_list(text: str) -> bool:
    marker_hits = len(_REFERENCE_MARKER_RE.findall(text))
    footnote_hits = len(_FOOTNOTE_MARK_RE.findall(text))
    return marker_hits >= 2 or footnote_hits >= 3 or (marker_hits + footnote_hits) >= 2


def semantic_chunk(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks, current, count = [], [], 0

    for sent in sentences:
        words = len(sent.split())
        if count + words > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += words

    if current:
        chunks.append(" ".join(current))

    cleaned = [c.strip() for c in chunks if len(c.strip()) > 20]
    return [c for c in cleaned if not _looks_like_reference_list(c)]
