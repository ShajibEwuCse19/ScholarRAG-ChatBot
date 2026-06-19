"""Grounded answer prompt construction."""

from __future__ import annotations

from src.utils.helpers import RetrievedChunk

SYSTEM_PROMPT = """You are ScholarRAG, a research-paper question answering assistant.
Answer only from the supplied excerpts. Do not use outside knowledge or invent facts.
Cite factual statements using the excerpt labels such as [1] or [2].
If the excerpts do not contain enough evidence, say: "I cannot answer from the indexed papers."
Keep the answer concise and explain technical terms when useful."""


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Build an Ollama chat request with numbered source excerpts."""

    context = "\n\n".join(
        f"[{index}] Paper: {chunk.paper_name} | Page: {chunk.page_number}\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Research excerpts:\n\n{context}\n\nQuestion: {question}",
        },
    ]


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Return a deterministic, deduplicated citation list."""

    seen: set[tuple[str, int]] = set()
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        key = (chunk.paper_name, chunk.page_number)
        if key not in seen:
            seen.add(key)
            lines.append(f"[{index}] {chunk.paper_name}, page {chunk.page_number}")
    return "\n".join(lines)
