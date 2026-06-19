# Agent Instructions

- Keep ScholarRAG fully local; do not add cloud LLM or paid API dependencies.
- Preserve paper name and 1-based page metadata through every retrieval path.
- Read runtime configuration from `src/config/settings.py` and avoid absolute paths.
- Keep modules focused, typed, documented, and understandable to beginner Python developers.
- Do not add LangChain unless the project owner explicitly requests it.
- Never commit `.env`, `prompt.txt`, PDFs, datasets, vector storage, uploads, or generated outputs.
- Run offline tests after changes and mock Ollama in automated tests.
