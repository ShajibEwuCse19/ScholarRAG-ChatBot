"""Citation-grounded retrieval-augmented generation pipeline."""

from __future__ import annotations

from collections.abc import Iterator

from src.llm.ollama_client import OllamaClient
from src.rag.prompt_template import build_messages, format_sources
from src.rag.retriever import Retriever


class RAGPipeline:
    """Retrieve paper evidence and stream a grounded local answer."""

    def __init__(self, retriever: Retriever, llm_client: OllamaClient) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def stream_answer(self, question: str) -> Iterator[str]:
        """Yield answer deltas followed by deterministic source citations."""

        chunks = self.retriever.retrieve(question)
        if not chunks:
            yield "No papers are indexed yet. Add and index at least one PDF."
            return

        messages = build_messages(question.strip(), chunks)
        yielded = False
        for delta in self.llm_client.stream_chat(messages):
            yielded = True
            yield delta
        if not yielded:
            yield "I cannot answer from the indexed papers."
        sources = format_sources(chunks)
        if sources:
            yield f"\n\n**Sources**\n{sources}"
