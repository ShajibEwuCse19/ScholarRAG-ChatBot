"""Reusable PDF ingestion workflow."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR, ensure_runtime_directories
from src.embeddings.embedding_model import EmbeddingModel
from src.pdf.pdf_loader import load_pdf
from src.preprocessing.chunker import chunk_pages
from src.utils.helpers import safe_filename, sha256_file
from src.vectordb.chroma_store import ChromaStore


@dataclass(frozen=True)
class IngestionResult:
    """Summary of one PDF ingestion attempt."""

    paper_name: str
    pages: int
    chunks: int
    skipped: bool = False


def ingest_pdf(
    path: str | Path,
    embedding_model: EmbeddingModel,
    store: ChromaStore,
) -> IngestionResult:
    """Extract, chunk, embed, and persist one PDF."""

    pages = load_pdf(path)
    document_hash = pages[0].document_hash
    if store.contains_document(document_hash):
        return IngestionResult(pages[0].paper_name, len(pages), 0, skipped=True)

    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError(f"No chunks could be created from {pages[0].paper_name}")
    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])
    store.delete_source(pages[0].source_path)
    store.upsert_chunks(chunks, embeddings)
    return IngestionResult(pages[0].paper_name, len(pages), len(chunks))


def find_pdfs(path: str | Path) -> list[Path]:
    """Return PDFs from a file or directory in stable order."""

    input_path = Path(path).expanduser().resolve()
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".pdf" else []
    if input_path.is_dir():
        return sorted(
            (candidate for candidate in input_path.rglob("*") if candidate.suffix.lower() == ".pdf"),
            key=lambda item: str(item).lower(),
        )
    return []


def save_uploaded_pdf(path: str | Path) -> Path:
    """Copy a UI upload to private local storage using a collision-safe name."""

    ensure_runtime_directories()
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {source.name}")
    digest = sha256_file(source)[:12]
    destination = UPLOAD_DIR / f"{digest}_{safe_filename(source.name)}"
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination
