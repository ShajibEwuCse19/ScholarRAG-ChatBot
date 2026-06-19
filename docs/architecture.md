# Architecture

ScholarRAG has a deliberately small local pipeline:

```text
PDF -> PyMuPDF pages -> boundary-aware chunks -> MiniLM embeddings -> ChromaDB
                                                                       |
Question -> MiniLM query embedding -> top-k chunks -> grounded prompt -+
                                                    -> Ollama -> streamed cited answer
```

## Components

- `src/pdf` extracts page text and immutable citation metadata.
- `src/preprocessing` creates bounded chunks without mixing pages.
- `src/embeddings` lazily loads the configured sentence-transformers model.
- `src/vectordb` owns persistent cosine-distance Chroma operations.
- `src/rag` retrieves evidence, builds the grounded prompt, and streams answers.
- `src/llm` is a small Ollama HTTP adapter with no cloud fallback.
- `src/evaluation` measures whether expected paper/page evidence appears in top-k retrieval.

Document SHA-256 hashes prevent duplicate ingestion. Chunk IDs include the document hash, page, position, and text. Chroma metadata stores the original paper name and 1-based page number used in citations.

Runtime PDFs, embeddings, evaluation data, and outputs live only in Git-ignored directories. Configuration is resolved from `.env` in `src/config/settings.py`; no absolute project path is hardcoded.
