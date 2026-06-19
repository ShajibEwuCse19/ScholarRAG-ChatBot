"""Persistent ChromaDB storage for ScholarRAG chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from src.config.settings import CHROMA_COLLECTION, CHROMA_DIR, ensure_runtime_directories
from src.utils.helpers import RetrievedChunk, TextChunk


class ChromaStore:
    """Store and query pre-computed local embeddings in ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION,
    ) -> None:
        ensure_runtime_directories()
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self.collection.count()

    def contains_document(self, document_hash: str) -> bool:
        """Return whether at least one chunk for a document is indexed."""

        result = self.collection.get(
            where={"document_hash": document_hash}, limit=1, include=[]
        )
        return bool(result.get("ids"))

    def delete_source(self, source_path: str) -> None:
        """Delete stale chunks previously indexed from a source path."""

        self.collection.delete(where={"source_path": source_path})

    def upsert_chunks(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> None:
        """Insert or update chunks and their embeddings."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal lengths")
        if not chunks:
            return
        metadatas: list[dict[str, Any]] = [
            {
                "paper_name": chunk.paper_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "source_path": chunk.source_path,
                "document_hash": chunk.document_hash,
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_embedding: list[float], n_results: int) -> list[RetrievedChunk]:
        """Return nearest chunks ordered by cosine distance."""

        if n_results <= 0 or self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                id=chunk_id,
                text=document or "",
                paper_name=str(metadata.get("paper_name", "Unknown")),
                page_number=int(metadata.get("page_number", 0)),
                chunk_index=int(metadata.get("chunk_index", 0)),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
