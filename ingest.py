"""Command-line PDF ingestion for ScholarRAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from src.config.settings import PAPERS_DIR, ensure_runtime_directories
from src.embeddings.embedding_model import EmbeddingModel
from src.ingestion import find_pdfs, ingest_pdf
from src.vectordb.chroma_store import ChromaStore


def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion command-line parser."""

    parser = argparse.ArgumentParser(description="Index research paper PDFs into ScholarRAG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PAPERS_DIR,
        help="A PDF or directory of PDFs (default: data/papers).",
    )
    return parser


def main() -> int:
    """Run ingestion and print a concise per-paper summary."""

    args = build_parser().parse_args()
    ensure_runtime_directories()
    pdfs = find_pdfs(args.input)
    if not pdfs:
        print(f"No PDF files found at: {args.input}", file=sys.stderr)
        return 1

    model = EmbeddingModel()
    store = ChromaStore()
    failures = 0
    for pdf_path in tqdm(pdfs, desc="Indexing papers", unit="paper"):
        try:
            result = ingest_pdf(pdf_path, model, store)
            if result.skipped:
                tqdm.write(f"Skipped duplicate: {result.paper_name}")
            else:
                tqdm.write(
                    f"Indexed {result.paper_name}: {result.pages} pages, {result.chunks} chunks"
                )
        except Exception as exc:
            failures += 1
            tqdm.write(f"Failed {pdf_path.name}: {exc}", file=sys.stderr)

    print(f"Collection contains {store.count()} chunks.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
