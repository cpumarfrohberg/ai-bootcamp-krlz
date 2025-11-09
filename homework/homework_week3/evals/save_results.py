"""Save evaluation results to CSV and metadata JSON"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def save_evaluation_results(
    results: list[dict[str, Any]],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save evaluation results as CSV and metadata JSON.

    Args:
        results: List of evaluation results, each containing:
            - question: str
            - hit_rate: float
            - mrr: float
            - judge_score: float
            - num_tokens: int
            - combined_score: float
        output_path: Path to output CSV file (should end with .csv)
        metadata: Optional metadata dict with evaluation parameters

    Returns:
        Path to the saved CSV file

    Example:
        results = [
            {
                "question": "What factors influence customer behavior?",
                "hit_rate": 1.0,
                "mrr": 1.0,
                "judge_score": 0.85,
                "num_tokens": 2500,
                "combined_score": 1.42,
            },
            ...
        ]

        metadata = {
            "model": "gpt-4o-mini",
            "judge_model": "gpt-4o",
            "num_questions": 20,
        }

        path = save_evaluation_results(
            results, "evals/results/evaluation.csv", metadata
        )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .csv extension
    if output_path.suffix != ".csv":
        output_path = output_path.with_suffix(".csv")

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Sort by combined_score descending (best results first)
    if "combined_score" in df.columns:
        df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # Save DataFrame as CSV
    df.to_csv(output_path, index=False)

    # Calculate aggregated metrics
    summary = {}
    if "hit_rate" in df.columns:
        summary["avg_hit_rate"] = float(df["hit_rate"].mean())
    if "mrr" in df.columns:
        summary["avg_mrr"] = float(df["mrr"].mean())
    if "judge_score" in df.columns:
        summary["avg_judge_score"] = float(df["judge_score"].mean())
    if "num_tokens" in df.columns:
        summary["avg_num_tokens"] = float(df["num_tokens"].mean())
        summary["total_tokens"] = int(df["num_tokens"].sum())
    if "combined_score" in df.columns:
        summary["avg_combined_score"] = float(df["combined_score"].mean())
        summary["best_combined_score"] = float(df["combined_score"].max())

    # Save metadata as JSON in same directory
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_data = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(results),
        "summary": summary,
    }
    if metadata:
        metadata_data["metadata"] = metadata

    with open(metadata_path, "w") as f:
        json.dump(metadata_data, f, indent=2)

    return output_path
