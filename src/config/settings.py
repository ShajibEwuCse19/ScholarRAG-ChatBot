"""Central configuration for ScholarRAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
EVALUATION_DATASET = DATA_DIR / "evaluation" / "qa_dataset.csv"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma_db"
UPLOAD_DIR = STORAGE_DIR / "uploaded_papers"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "scholar_rag_papers")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5"))
OLLAMA_READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "300"))

GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))


def ensure_runtime_directories() -> None:
    """Create local runtime directories when they do not exist."""

    for directory in (PAPERS_DIR, CHROMA_DIR, UPLOAD_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
