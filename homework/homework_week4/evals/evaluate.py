"""Main evaluation runner for Wikipedia agent"""

import json
import logging
from pathlib import Path
from typing import Any

from config import DEFAULT_SEARCH_MODE, SearchMode
from evals.combined_score import calculate_combined_score
from evals.judge import evaluate_answer
from evals.save_results import save_evaluation_results
from evals.source_metrics import calculate_hit_rate, calculate_mrr
from wikiagent.wikipagent import query_wikipedia

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "evals/results/evaluation.json"
QUESTION_PREVIEW_LENGTH = 50  # Max length for question in logs
SCORE_DECIMAL_PLACES = 2  # Decimal places for score formatting
FALLBACK_HIT_RATE = 0.0  # Fallback hit rate for failed evaluations
FALLBACK_MRR = 0.0  # Fallback MRR for failed evaluations
FALLBACK_JUDGE_SCORE = 0.0  # Fallback judge score for failed evaluations
FALLBACK_NUM_TOKENS = 0  # Fallback token count for failed evaluations
FALLBACK_COMBINED_SCORE = 0.0  # Fallback combined score for failed evaluations


async def evaluate_agent(
    ground_truth_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    search_mode: SearchMode = DEFAULT_SEARCH_MODE,
    judge_model: str | None = None,
) -> Path:
    """
    Run full evaluation workflow on Wikipedia agent.

    Args:
        ground_truth_path: Path to ground truth JSON file
        output_path: Path to output JSON file (default: DEFAULT_OUTPUT_PATH)
        search_mode: Search mode for agent (default: EVALUATION)
        judge_model: Model to use for judging (default: from config)

    Returns:
        Path to saved JSON file

    Example:
        path = await evaluate_agent(
            "evals/ground_truth.json",
            "evals/results/evaluation.json",
            SearchMode.EVALUATION,
        )
    """
    ground_truth_path = Path(ground_truth_path)
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    logger.info(f"Loaded {len(ground_truth)} questions from ground truth")

    results = []

    # Evaluate each question
    for i, item in enumerate(ground_truth, 1):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])

        logger.info(
            f"Evaluating question {i}/{len(ground_truth)}: {question[:QUESTION_PREVIEW_LENGTH]}..."
        )

        try:
            agent_result = await query_wikipedia(question, search_mode=search_mode)

            actual_sources = agent_result.answer.sources_used or []
            hit_rate = calculate_hit_rate(expected_sources, actual_sources)
            mrr = calculate_mrr(expected_sources, actual_sources)

            judge_result = await evaluate_answer(
                question,
                agent_result.answer,
                tool_calls=agent_result.tool_calls,
                judge_model=judge_model,
            )
            judge_score = judge_result.evaluation.overall_score

            # Count tokens (agent + judge)
            agent_tokens = agent_result.usage.total_tokens
            judge_tokens = judge_result.usage.total_tokens
            total_tokens = agent_tokens + judge_tokens

            # Calculate combined score
            combined_score = calculate_combined_score(
                hit_rate=hit_rate,
                judge_score=judge_score,
                num_tokens=total_tokens,
            )

            results.append(
                {
                    "question": question,
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "judge_score": judge_score,
                    "num_tokens": total_tokens,
                    "combined_score": combined_score,
                }
            )

            score_format = f".{SCORE_DECIMAL_PLACES}f"
            logger.info(
                f"Question {i} complete: hit_rate={hit_rate:{score_format}}, "
                f"mrr={mrr:{score_format}}, judge_score={judge_score:{score_format}}, "
                f"combined_score={combined_score:{score_format}}"
            )

        except Exception as e:
            logger.error(f"Error evaluating question {i}: {e}")
            # Store failed result with fallback values
            results.append(
                {
                    "question": question,
                    "hit_rate": FALLBACK_HIT_RATE,
                    "mrr": FALLBACK_MRR,
                    "judge_score": FALLBACK_JUDGE_SCORE,
                    "num_tokens": FALLBACK_NUM_TOKENS,
                    "combined_score": FALLBACK_COMBINED_SCORE,
                }
            )

    # Prepare metadata
    metadata: dict[str, Any] = {
        "ground_truth_file": str(ground_truth_path),
        "search_mode": str(search_mode),
    }
    if judge_model:
        metadata["judge_model"] = judge_model

    output_path = save_evaluation_results(results, output_path, metadata)

    logger.info(f"Evaluation complete. Results saved to: {output_path}")

    return output_path
