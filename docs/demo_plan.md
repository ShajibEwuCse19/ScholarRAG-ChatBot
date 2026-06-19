# Demo Plan

1. Start Ollama and confirm `llama3.2:3b` is installed.
2. Activate `.venv` and run `python app.py`.
3. Upload one or more English research PDFs and select **Save and index PDFs**.
4. Ask a question whose answer is visible in a known paper section.
5. Confirm the response streams and includes paper/page citations.
6. Ask an unsupported question and confirm the assistant declines to invent an answer.
7. Prepare the private evaluation CSV and run `python evaluate.py`.
8. Show Hit@4, MRR, and the ignored detailed result file.

Before demonstrating, keep PDFs small enough for quick indexing and verify the Ollama model responds locally. Do not enable a public Gradio share link because the project is designed for private local use.
