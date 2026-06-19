"""Offline tests for ScholarRAG's core behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz
import pandas as pd
import requests

from src.evaluation.evaluator import evaluate_retrieval
from src.ingestion import ingest_pdf
from src.llm.ollama_client import OllamaClient, OllamaError
from src.pdf.pdf_loader import PDFLoadError, load_pdf
from src.preprocessing.chunker import chunk_pages
from src.rag.prompt_template import build_messages, format_sources
from src.rag.rag_pipeline import RAGPipeline
from src.utils.helpers import PDFPage, RetrievedChunk
from src.vectordb.chroma_store import ChromaStore


class FakeEmbeddingModel:
    """Deterministic embeddings that avoid model downloads in tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float("method" in text.lower()), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float("method" in text.lower()), 1.0]


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievedChunk]:
        return self.results[:top_k]


class FakeLLM:
    def stream_chat(self, messages: list[dict[str, str]]):
        yield "The method improves retrieval [1]."


class FailingSession:
    def post(self, *args, **kwargs):
        raise requests.ConnectionError("offline")


def make_pdf(path: Path) -> None:
    """Create a small two-page PDF fixture."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Introduction. This paper studies retrieval methods.")
    page = document.new_page()
    page.insert_text((72, 72), "Results. The method improves citation accuracy.")
    document.save(path)
    document.close()


class ScholarRAGTests(unittest.TestCase):
    def test_missing_pdf_has_clear_error(self) -> None:
        with self.assertRaisesRegex(PDFLoadError, "PDF not found"):
            load_pdf("missing-paper.pdf")

    def test_pdf_loading_and_page_metadata(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory) / "paper.pdf"
            make_pdf(path)
            pages = load_pdf(path)
        self.assertEqual([page.page_number for page in pages], [1, 2])
        self.assertEqual(pages[0].paper_name, "paper.pdf")
        self.assertIn("retrieval", pages[0].text)

    def test_chunking_is_page_safe_and_overlapping(self) -> None:
        page = PDFPage(
            "paper.pdf",
            3,
            "First sentence contains several words. Second sentence adds evidence. "
            "Third sentence concludes the discussion.",
            "paper.pdf",
            "hash",
        )
        chunks = chunk_pages([page], chunk_size=70, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page_number == 3 for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 70 for chunk in chunks))
        self.assertTrue(set(chunks[0].text.split()) & set(chunks[1].text.split()))

    def test_ingestion_skips_duplicate_and_queries(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            pdf = root / "paper.pdf"
            make_pdf(pdf)
            store = ChromaStore(root / "chroma", "test_collection")
            model = FakeEmbeddingModel()
            first = ingest_pdf(pdf, model, store)  # type: ignore[arg-type]
            second = ingest_pdf(pdf, model, store)  # type: ignore[arg-type]
            results = store.query([1.0, 1.0], 2)
        self.assertGreater(first.chunks, 0)
        self.assertTrue(second.skipped)
        self.assertTrue(results)
        self.assertEqual(results[0].paper_name, "paper.pdf")

    def test_prompt_and_pipeline_include_sources(self) -> None:
        chunk = RetrievedChunk("1", "Evidence", "paper.pdf", 2, 0, 0.1)
        messages = build_messages("What happened?", [chunk])
        self.assertIn("Paper: paper.pdf | Page: 2", messages[1]["content"])
        self.assertEqual(format_sources([chunk]), "[1] paper.pdf, page 2")
        pipeline = RAGPipeline(FakeRetriever([chunk]), FakeLLM())  # type: ignore[arg-type]
        answer = "".join(pipeline.stream_answer("What happened?"))
        self.assertIn("retrieval", answer)
        self.assertIn("**Sources**", answer)

    def test_pipeline_handles_empty_index_without_calling_llm(self) -> None:
        pipeline = RAGPipeline(FakeRetriever([]), FakeLLM())  # type: ignore[arg-type]
        answer = "".join(pipeline.stream_answer("What happened?"))
        self.assertIn("No papers are indexed", answer)

    def test_ollama_connection_error_is_actionable(self) -> None:
        client = OllamaClient(session=FailingSession())  # type: ignore[arg-type]
        with self.assertRaisesRegex(OllamaError, "Ollama request failed"):
            list(client.stream_chat([{"role": "user", "content": "Hello"}]))

    def test_retrieval_evaluation_metrics(self) -> None:
        result = RetrievedChunk("1", "Evidence", "paper.pdf", 2, 0, 0.1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "qa.csv"
            output = root / "results.csv"
            pd.DataFrame(
                [{"question": "Question?", "expected_paper": "paper.pdf", "expected_page": 2}]
            ).to_csv(dataset, index=False)
            summary = evaluate_retrieval(
                dataset, output, FakeRetriever([result]), top_k=4  # type: ignore[arg-type]
            )
            self.assertTrue(output.exists())
        self.assertEqual(summary.hit_rate, 1.0)
        self.assertEqual(summary.mrr, 1.0)

    def test_evaluation_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "invalid.csv"
            pd.DataFrame([{"question": "Question?"}]).to_csv(dataset, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                evaluate_retrieval(
                    dataset,
                    root / "results.csv",
                    FakeRetriever([]),  # type: ignore[arg-type]
                    top_k=4,
                )


if __name__ == "__main__":
    unittest.main()
