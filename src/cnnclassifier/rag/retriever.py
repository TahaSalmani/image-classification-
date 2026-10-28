"""TF-IDF retrieval over the knowledge base (no embedding model, no API).

The corpus is a dozen short, hand-written passages, so a sparse TF-IDF
vectoriser is enough: deterministic, no GPU, no download, and roughly 30 MB
of RAM on top of Flask. ``sentence-transformers`` would cost hundreds of
megabytes, which a 512 MB Render instance cannot spare next to TensorFlow.

Two normalisations carry the recall, because plain TF-IDF has no notion of
morphology:

* ``x-ray`` / ``x ray`` / ``xray`` collapse to one token
* a suffix stripper maps "cause", "causes", "caused" and "causing" to the same
  term, so a question phrased with a different inflection still finds the
  passage
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk, load_knowledge_base

_XRAY_RE = re.compile(r"\bx[\s\-]*ray", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def normalize(text: str) -> str:
    """Lowercase and merge the spellings of X-ray."""
    return _XRAY_RE.sub("xray", text.lower())


def stem(word: str) -> str:
    """Strip the handful of English suffixes that matter on this corpus.

    Deliberately crude: it only has to agree with itself, so "cause",
    "causes", "caused" and "causing" all end up as "caus".
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def analyze(text: str) -> list[str]:
    """Unigrams plus bigrams of stemmed, non-stopword tokens."""
    tokens = [
        stem(token)
        for token in _TOKEN_RE.findall(normalize(text))
        if len(token) > 1 and token not in _STOP_WORDS
    ]
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


@dataclass(frozen=True)
class Hit:
    """A retrieved passage plus the similarity that earned it its place."""

    chunk: Chunk
    score: float

    def as_dict(self) -> dict:
        return {**self.chunk.as_dict(), "score": round(self.score, 4)}


class KnowledgeRetriever:
    """Rank knowledge-base passages against a question."""

    def __init__(self, chunks: list[Chunk], *, min_df: int = 1) -> None:
        if not chunks:
            raise ValueError("cannot build a retriever from zero passages")
        self.chunks = list(chunks)
        # sublinear_tf damps the long bullet lists so one verbose passage
        # cannot dominate every query.
        self._vectorizer = TfidfVectorizer(
            analyzer=analyze,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(
            [chunk.text for chunk in self.chunks]
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> list[Hit]:
        """Return up to ``k`` passages scoring at least ``min_score``."""
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        query_matrix = self._vectorizer.transform([query])
        scores = cosine_similarity(query_matrix, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits = []
        for index in ranked[:k]:
            score = float(scores[index])
            if score < min_score:
                break
            hits.append(Hit(chunk=self.chunks[index], score=score))
        return hits


_RETRIEVER: KnowledgeRetriever | None = None
_LOCK = threading.Lock()


def get_retriever(directory: Path | str | None = None) -> KnowledgeRetriever:
    """Build the retriever once per process (gunicorn loads it per worker)."""
    global _RETRIEVER
    if _RETRIEVER is None:
        with _LOCK:
            if _RETRIEVER is None:
                _RETRIEVER = KnowledgeRetriever(load_knowledge_base(directory))
    return _RETRIEVER


def reset_retriever() -> None:
    """Drop the cached retriever (used by tests and by ``KNOWLEDGE_DIR`` changes)."""
    global _RETRIEVER
    with _LOCK:
        _RETRIEVER = None


__all__ = [
    "Hit",
    "KnowledgeRetriever",
    "analyze",
    "get_retriever",
    "normalize",
    "reset_retriever",
    "stem",
]
