"""Gradio application entry point for ScholarRAG."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

from src.config.settings import GRADIO_HOST, GRADIO_PORT, ensure_runtime_directories
from src.embeddings.embedding_model import EmbeddingModel
from src.ingestion import ingest_pdf, save_uploaded_pdf
from src.llm.ollama_client import OllamaClient, OllamaError
from src.rag.rag_pipeline import RAGPipeline
from src.rag.retriever import Retriever
from src.vectordb.chroma_store import ChromaStore


def create_app() -> gr.Blocks:
    """Create the ScholarRAG Gradio interface and local services."""

    ensure_runtime_directories()
    embedding_model = EmbeddingModel()
    store = ChromaStore()
    pipeline = RAGPipeline(Retriever(embedding_model, store), OllamaClient())

    def index_uploads(files: list[str] | None) -> str:
        """Persist and index PDF uploads, returning a UI status message."""

        if not files:
            return "Select one or more PDF files first."
        messages: list[str] = []
        for file_value in files:
            try:
                saved = save_uploaded_pdf(Path(file_value))
                result = ingest_pdf(saved, embedding_model, store)
                if result.skipped:
                    messages.append(f"Skipped duplicate: {result.paper_name}")
                else:
                    messages.append(
                        f"Indexed {result.paper_name}: {result.pages} pages, {result.chunks} chunks"
                    )
            except Exception as exc:
                messages.append(f"Failed {Path(file_value).name}: {exc}")
        messages.append(f"Collection total: {store.count()} chunks")
        return "\n".join(messages)

    def respond(
        message: str, history: list[dict[str, Any]] | None
    ) -> Iterator[tuple[list[dict[str, Any]], str]]:
        """Stream a pipeline answer into Gradio message history."""

        history = list(history or [])
        question = message.strip()
        if not question:
            yield history, ""
            return
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ""},
            ]
        )
        yield history, ""
        answer = ""
        try:
            for delta in pipeline.stream_answer(question):
                answer += delta
                history[-1] = {"role": "assistant", "content": answer}
                yield history, ""
        except OllamaError as exc:
            history[-1] = {"role": "assistant", "content": f"Ollama error: {exc}"}
            yield history, ""
        except Exception as exc:
            history[-1] = {"role": "assistant", "content": f"Request failed: {exc}"}
            yield history, ""

    with gr.Blocks(title="ScholarRAG") as demo:
        gr.Markdown(
            "# ScholarRAG\nAsk questions about English research papers using a fully local, cited RAG pipeline."
        )
        with gr.Row():
            uploads = gr.File(
                label="Research paper PDFs",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
            )
            with gr.Column():
                index_button = gr.Button("Save and index PDFs", variant="primary")
                index_status = gr.Textbox(label="Indexing status", interactive=False, lines=5)
        chatbot = gr.Chatbot(label="Paper chat", height=450)
        question = gr.Textbox(
            label="Question",
            placeholder="What do the indexed papers say about ...?",
        )
        with gr.Row():
            send = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear chat")

        index_button.click(index_uploads, inputs=uploads, outputs=index_status)
        send.click(respond, inputs=[question, chatbot], outputs=[chatbot, question])
        question.submit(respond, inputs=[question, chatbot], outputs=[chatbot, question])
        clear.click(lambda: ([], ""), outputs=[chatbot, question])
    return demo


def main() -> None:
    """Launch ScholarRAG on the configured local interface."""

    create_app().launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT)


if __name__ == "__main__":
    main()
