"""Split the sourced knowledge base into small, quotable passages.

Contract for every file in ``knowledge/``:

* line 1 is ``SOURCE: <url>`` (a file may carry several ``SOURCE:`` lines)
* the body is copied from that source
* the last line is the attribution / licence notice

This module never rewrites the medical text. It only strips the provenance
lines out of the searchable passage and keeps them as metadata, then groups
paragraphs so each passage stays short enough to quote verbatim.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KB_DIR = Path(__file__).resolve().parents[3] / "knowledge"

#: Target size for a passage. Blocks are only split at paragraph or bullet-list
#: boundaries, so a single dense block (the risk-factor list) may exceed it a
#: little; on this corpus 500 characters keeps retrieval focused without
#: cutting a paragraph in half.
MAX_CHUNK_CHARS = 500

SOURCE_RE = re.compile(r"^SOURCE\s*:\s*(\S+)\s*$", re.IGNORECASE)

#: Lines that credit the source rather than describe pneumonia. They are
#: kept out of the TF-IDF text so licence boilerplate cannot outrank content.
ATTRIBUTION_PREFIXES = (
    "contains public sector information",
    "source: cdc",
    "source: medlineplus",
    "sources: medlineplus",
    "source: nhs",
)


@dataclass(frozen=True)
class Chunk:
    """One passage from the knowledge base."""

    chunk_id: str
    file: str
    sources: tuple[str, ...]
    text: str
    attribution: str

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file": self.file,
            "sources": list(self.sources),
            "text": self.text,
            "attribution": self.attribution,
        }


def knowledge_dir() -> Path:
    """Knowledge base location, overridable with ``KNOWLEDGE_DIR``."""
    return Path(os.environ.get("KNOWLEDGE_DIR") or DEFAULT_KB_DIR)


def _split_blocks(body: str) -> list[str]:
    blocks = []
    current: list[str] = []
    for line in body.splitlines():
        if line.strip():
            current.append(line.rstrip())
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(blocks: list[str], max_chars: int) -> list[str]:
    """Merge consecutive blocks, breaking before ``max_chars`` is exceeded."""
    packed: list[str] = []
    buffer = ""
    for block in blocks:
        candidate = f"{buffer}\n\n{block}" if buffer else block
        if buffer and len(candidate) > max_chars:
            packed.append(buffer)
            buffer = block
        else:
            buffer = candidate
    if buffer:
        packed.append(buffer)
    return packed


def parse_file(path: Path) -> list[Chunk]:
    """Parse one markdown file into passages. Raises if ``SOURCE:`` is missing."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    lines = raw.splitlines()
    sources: list[str] = []
    body_lines: list[str] = []
    attribution_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        match = SOURCE_RE.match(stripped)
        if match:
            sources.append(match.group(1))
        elif stripped.lower().startswith(ATTRIBUTION_PREFIXES):
            attribution_lines.append(stripped)
        else:
            body_lines.append(line)

    if not sources:
        raise ValueError(f"{path.name}: first line must be 'SOURCE: <url>'")

    blocks = _split_blocks("\n".join(body_lines))
    if not blocks:
        return []

    attribution = " ".join(attribution_lines)
    return [
        Chunk(
            chunk_id=f"{path.stem}#{index}",
            file=path.name,
            sources=tuple(sources),
            text=text,
            attribution=attribution,
        )
        for index, text in enumerate(_pack(blocks, MAX_CHUNK_CHARS))
    ]


def load_knowledge_base(directory: Path | str | None = None) -> list[Chunk]:
    """Load every ``*.md`` file in the knowledge directory, sorted by name."""
    root = Path(directory) if directory is not None else knowledge_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"knowledge directory not found: {root}")

    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        chunks.extend(parse_file(path))
    if not chunks:
        raise ValueError(f"no passages loaded from {root}")
    return chunks
