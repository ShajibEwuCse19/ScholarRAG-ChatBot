"""Boundary-aware text chunking for extracted PDF pages."""

from __future__ import annotations

import hashlib
import re

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from src.utils.helpers import PDFPage, TextChunk


def _units(text: str) -> list[str]:
    """Split text into paragraphs and sentences without external NLP models."""

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip())
    return units


def _split_oversized(unit: str, chunk_size: int) -> list[str]:
    """Split a long sentence on words so no chunk is unbounded."""

    words = unit.split()
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and length + added > chunk_size:
            pieces.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += added
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_suffix(text: str, overlap: int) -> str:
    """Return a word-aligned suffix bounded by the overlap target."""

    if overlap <= 0:
        return ""
    words = text.split()
    selected: list[str] = []
    length = 0
    for word in reversed(words):
        added = len(word) + (1 if selected else 0)
        if selected and length + added > overlap:
            break
        selected.append(word)
        length += added
    return " ".join(reversed(selected))


def chunk_pages(
    pages: list[PDFPage],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Create citation-safe chunks that never cross page boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        expanded: list[str] = []
        unit_limit = max(1, chunk_size - overlap)
        for unit in _units(page.text):
            expanded.extend(_split_oversized(unit, unit_limit))

        current = ""
        page_chunks: list[str] = []
        for unit in expanded:
            candidate = f"{current} {unit}".strip()
            if current and len(candidate) > chunk_size:
                page_chunks.append(current)
                prefix = _overlap_suffix(current, overlap)
                current = f"{prefix} {unit}".strip()
            else:
                current = candidate
        if current:
            page_chunks.append(current)

        for chunk_index, text in enumerate(page_chunks):
            identity = f"{page.document_hash}:{page.page_number}:{chunk_index}:{text}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    text=text,
                    paper_name=page.paper_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    source_path=page.source_path,
                    document_hash=page.document_hash,
                )
            )
    return chunks
