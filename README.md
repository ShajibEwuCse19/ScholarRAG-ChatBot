# ScholarRAG

A fully local, citation-grounded RAG chatbot for asking questions about English research papers. PDF text, embeddings, vector search, and answer generation remain on your machine.

## Features

- Extracts PDF text with 1-based page metadata using PyMuPDF.
- Creates coherent, overlapping chunks that never cross page boundaries.
- Embeds locally with `sentence-transformers/all-MiniLM-L6-v2`.
- Persists vectors in a local ChromaDB collection.
- Streams grounded answers from a local Ollama model.
- Shows deterministic paper-and-page citations.
- Accepts persistent local PDF uploads through Gradio.
- Evaluates retrieval with Hit@K and mean reciprocal rank (MRR).

No paid API, cloud LLM, OpenAI API, Gemini API, or LangChain is used.

## Prerequisites

- Python 3.12 (the current environment uses Python 3.12.10)
- [Ollama](https://ollama.com/) for answer generation

Install Ollama separately, start it, and download the default model:

```powershell
ollama pull llama3.2:3b
```

Ingestion and retrieval work without Ollama. Only answer generation requires it.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The embedding model downloads on first use and is then available from the local model cache. Change supported settings in `.env`; all defaults are documented in `.env.example`.

## Use

Place PDFs in `data/papers`, then ingest them:

```powershell
python ingest.py
```

You can also ingest one PDF or another directory:

```powershell
python ingest.py --input C:\path\to\paper.pdf
```

Start the private local UI:

```powershell
python app.py
```

Open `http://127.0.0.1:7860`. Uploaded PDFs are copied to `storage/uploaded_papers` and indexed persistently. Both the files and vectors are excluded from Git.

## Retrieval evaluation

Create the private file `data/evaluation/qa_dataset.csv` with this schema:

```csv
question,expected_paper,expected_page
What method is proposed?,example.pdf,3
```

The expected filename comparison is case-insensitive and the page is 1-based. Run:

```powershell
python evaluate.py
```

Results are printed and written to `outputs/results/retrieval_results.csv`. Both the dataset and generated result are ignored by Git.

## Tests

Tests are offline and use generated PDFs, deterministic fake embeddings, temporary Chroma collections, and a fake LLM:

```powershell
python -m unittest discover -s tests -v
```

## Privacy

`.gitignore` excludes `.env`, `prompt.txt`, virtual environments, PDFs, evaluation data, uploads, Chroma data, logs, screenshots, and generated results. Review `git status` before every push and never place secrets in tracked source files.
