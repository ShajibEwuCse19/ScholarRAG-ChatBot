"""Simple source-and-page retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.rag.retriever import Retriever

REQUIRED_COLUMNS = {"question", "expected_paper", "expected_page"}


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate retrieval metrics."""

    questions: int
    hits: int
    hit_rate: float
    mrr: float


def _normalize_paper(value: object) -> str:
    return Path(str(value).strip()).name.casefold()


def evaluate_retrieval(
    dataset_path: str | Path,
    output_path: str | Path,
    retriever: Retriever,
    top_k: int,
) -> EvaluationSummary:
    """Evaluate exact expected paper/page retrieval and write per-query results."""

    dataset = Path(dataset_path)
    if not dataset.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset}")
    frame = pd.read_csv(dataset)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Evaluation CSV contains no questions")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    rows: list[dict[str, object]] = []
    reciprocal_ranks: list[float] = []
    for row_number, row in frame.iterrows():
        question = str(row["question"]).strip()
        expected_paper = _normalize_paper(row["expected_paper"])
        if not question or not expected_paper or pd.isna(row["expected_page"]):
            raise ValueError(f"Invalid evaluation row {row_number + 2}")
        expected_page = int(row["expected_page"])
        results = retriever.retrieve(question, top_k=top_k)
        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if _normalize_paper(result.paper_name) == expected_paper
                and result.page_number == expected_page
            ),
            None,
        )
        reciprocal_rank = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)
        rows.append(
            {
                "question": question,
                "expected_paper": str(row["expected_paper"]),
                "expected_page": expected_page,
                "hit": bool(rank),
                "rank": rank or "",
                "reciprocal_rank": reciprocal_rank,
                "retrieved_sources": "; ".join(
                    f"{item.paper_name}:p{item.page_number}" for item in results
                ),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    hits = sum(bool(row["hit"]) for row in rows)
    count = len(rows)
    return EvaluationSummary(
        questions=count,
        hits=hits,
        hit_rate=hits / count,
        mrr=sum(reciprocal_ranks) / count,
    )
