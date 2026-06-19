"""PDF text extraction with page-level citation metadata."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from src.utils.helpers import PDFPage, sha256_file


class PDFLoadError(RuntimeError):
    """Raised when a PDF cannot be opened or read."""


def _clean_page_text(text: str) -> str:
    """Normalize PDF text while retaining paragraph boundaries."""

    blocks = re.split(r"\n\s*\n", text.replace("\x00", " "))
    cleaned = []
    for block in blocks:
        lines = [re.sub(r"\s+", " ", line).strip() for line in block.splitlines()]
        paragraph = " ".join(line for line in lines if line)
        if paragraph:
            cleaned.append(paragraph)
    return "\n\n".join(cleaned)


def load_pdf(path: str | Path) -> list[PDFPage]:
    """Extract non-empty pages from a PDF using 1-based page numbers."""

    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.is_file():
        raise PDFLoadError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Expected a PDF file: {pdf_path}")

    document_hash = sha256_file(pdf_path)
    try:
        with fitz.open(pdf_path) as document:
            if document.needs_pass:
                raise PDFLoadError(f"Encrypted PDF is not supported: {pdf_path.name}")
            pages = []
            for page_index, page in enumerate(document):
                text = _clean_page_text(page.get_text("text"))
                if text:
                    pages.append(
                        PDFPage(
                            paper_name=pdf_path.name,
                            page_number=page_index + 1,
                            text=text,
                            source_path=str(pdf_path),
                            document_hash=document_hash,
                        )
                    )
    except PDFLoadError:
        raise
    except Exception as exc:
        raise PDFLoadError(f"Could not read {pdf_path.name}: {exc}") from exc

    if not pages:
        raise PDFLoadError(f"No extractable text found in {pdf_path.name}")
    return pages
