"""Local sentence-transformers embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from src.config.settings import EMBEDDING_MODEL


class EmbeddingModel:
    """Lazily load and use a local sentence-transformers model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Load the model on first use and reuse it afterward."""

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized embeddings for document text."""

        if not texts:
            return []
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Return one normalized query embedding."""

        if not text.strip():
            raise ValueError("Question cannot be empty")
        return self.embed_documents([text])[0]
