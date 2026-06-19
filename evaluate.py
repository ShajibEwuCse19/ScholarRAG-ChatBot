"""Command-line retrieval evaluation for ScholarRAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config.settings import EVALUATION_DATASET, RESULTS_DIR, TOP_K
from src.embeddings.embedding_model import EmbeddingModel
from src.evaluation.evaluator import evaluate_retrieval
from src.rag.retriever import Retriever
from src.vectordb.chroma_store import ChromaStore


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""

    parser = argparse.ArgumentParser(description="Evaluate ScholarRAG retrieval.")
    parser.add_argument("--dataset", type=Path, default=EVALUATION_DATASET)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "retrieval_results.csv"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser


def main() -> int:
    """Run retrieval evaluation and print Hit@K and MRR."""

    args = build_parser().parse_args()
    try:
        retriever = Retriever(EmbeddingModel(), ChromaStore())
        summary = evaluate_retrieval(
            args.dataset, args.output, retriever, args.top_k
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Questions: {summary.questions}")
    print(f"Hits@{args.top_k}: {summary.hits}")
    print(f"Hit rate@{args.top_k}: {summary.hit_rate:.3f}")
    print(f"MRR: {summary.mrr:.3f}")
    print(f"Detailed results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
