"""Shared data records and small filesystem helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PDFPage:
    """Text extracted from one page of a PDF."""

    paper_name: str
    page_number: int
    text: str
    source_path: str
    document_hash: str


@dataclass(frozen=True)
class TextChunk:
    """A bounded text chunk and its citation metadata."""

    id: str
    text: str
    paper_name: str
    page_number: int
    chunk_index: int
    source_path: str
    document_hash: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by vector search."""

    id: str
    text: str
    paper_name: str
    page_number: int
    chunk_index: int
    distance: float


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe PDF filename."""

    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return f"{stem or 'paper'}.pdf"
