"""Question-to-chunk retrieval."""

from __future__ import annotations

from src.config.settings import TOP_K
from src.embeddings.embedding_model import EmbeddingModel
from src.utils.helpers import RetrievedChunk
from src.vectordb.chroma_store import ChromaStore


class Retriever:
    """Embed a question and retrieve nearest Chroma chunks."""

    def __init__(self, embedding_model: EmbeddingModel, store: ChromaStore) -> None:
        self.embedding_model = embedding_model
        self.store = store

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a non-empty question."""

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if self.store.count() == 0:
            return []
        vector = self.embedding_model.embed_query(question)
        return self.store.query(vector, top_k)
