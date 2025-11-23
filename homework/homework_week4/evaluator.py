"""Evaluate agent logs from logs/ directory.

Evaluates logs based on:
- follow_instruction: Did the agent follow its instructions?
- answer_relevant: Is the answer relevant to the question?

Saves results to Postgres EvalCheck table.
"""

import json
import logging
import re
from pathlib import Path

from wikiagent.monitoring.db import get_db, insert_eval_check
from wikiagent.monitoring.schemas import EvalCheckCreate

logger = logging.getLogger(__name__)

IDEAL_MIN_SEARCHES = 11
IDEAL_MAX_SEARCHES = 17
ACCEPTABLE_MIN_SEARCHES = 8
ACCEPTABLE_MAX_SEARCHES = 20

IDEAL_MIN_RETRIEVALS = 5
IDEAL_MAX_RETRIEVALS = 10
ACCEPTABLE_MIN_RETRIEVALS = 3
ACCEPTABLE_MAX_RETRIEVALS = 12

PERFECT_SCORE = 1.0
GOOD_SCORE = 0.7
POOR_SCORE = 0.3
ZERO_SCORE = 0.0
DEFAULT_OVERLAP_SCORE = 0.5

PASS_THRESHOLD = 0.7
AVERAGE_DIVISOR = 2

TOOL_CALL_KIND = "tool-call"
TOOL_WIKIPEDIA_SEARCH = "wikipedia_search"
TOOL_WIKIPEDIA_GET_PAGE = "wikipedia_get_page"


def read_logs(logs_dir: str = "logs/") -> list[dict]:
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        logger.error(f"Logs directory not found: {logs_dir}")
        return []

    all_logs = []
    for json_file in logs_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_logs.extend(data)
                elif isinstance(data, dict) and "logs" in data:
                    all_logs.extend(data["logs"])
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")

    return all_logs


def _is_tool_call(part: dict) -> bool:
    """Check if part is a tool call."""
    return part.get("part_kind") == TOOL_CALL_KIND


def _get_tool_name(part: dict) -> str | None:
    """Extract tool name from part."""
    return part.get("tool_name")


def count_tool_calls(messages: list[dict]) -> tuple[int, int]:
    searches = 0
    retrievals = 0

    for msg in messages:
        for part in msg.get("parts", []):
            if not _is_tool_call(part):
                continue

            tool_name = _get_tool_name(part)
            if tool_name == TOOL_WIKIPEDIA_SEARCH:
                searches += 1
            elif tool_name == TOOL_WIKIPEDIA_GET_PAGE:
                retrievals += 1

    return searches, retrievals


def parse_answer(answer_str: str) -> dict:
    result = {"answer": None, "sources": []}
    if not answer_str:
        return result

    match = re.search(r"answer=['\"](.*?)['\"]", answer_str, re.DOTALL)
    if match:
        result["answer"] = match.group(1)

    match = re.search(r"sources_used=\[(.*?)\]", answer_str)
    if match:
        result["sources"] = re.findall(r"['\"](.*?)['\"]", match.group(1))

    return result


def evaluate_follow_instruction(
    searches: int, retrievals: int, has_answer: bool
) -> float:
    """Score 0.0-1.0 based on instruction following."""
    scores = []

    # Search count
    if IDEAL_MIN_SEARCHES <= searches <= IDEAL_MAX_SEARCHES:
        scores.append(PERFECT_SCORE)
    elif ACCEPTABLE_MIN_SEARCHES <= searches <= ACCEPTABLE_MAX_SEARCHES:
        scores.append(GOOD_SCORE)
    else:
        scores.append(POOR_SCORE)

    # Retrieval count
    if IDEAL_MIN_RETRIEVALS <= retrievals <= IDEAL_MAX_RETRIEVALS:
        scores.append(PERFECT_SCORE)
    elif ACCEPTABLE_MIN_RETRIEVALS <= retrievals <= ACCEPTABLE_MAX_RETRIEVALS:
        scores.append(GOOD_SCORE)
    else:
        scores.append(POOR_SCORE)

    # Has answer
    scores.append(PERFECT_SCORE if has_answer else ZERO_SCORE)

    return sum(scores) / len(scores)


def _is_source_relevant(source: str, keywords: set[str]) -> bool:
    source_lower = source.lower()
    return any(keyword in source_lower for keyword in keywords)


def evaluate_answer_relevant(question: str, answer: str, sources: list[str]) -> float:
    if not answer:
        return ZERO_SCORE

    # Keyword overlap
    q_words = set(re.findall(r"\b\w+\b", question.lower())) - {
        "where",
        "do",
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
    }
    a_words = set(re.findall(r"\b\w+\b", answer.lower()))
    overlap = (
        len(q_words & a_words) / len(q_words) if q_words else DEFAULT_OVERLAP_SCORE
    )

    # Source relevance
    relevant_count = sum(
        1 for source in sources if _is_source_relevant(source, q_words)
    )
    source_score = relevant_count / len(sources) if sources else ZERO_SCORE

    return (overlap + source_score) / AVERAGE_DIVISOR


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    logs = read_logs()
    if not logs:
        logger.error("No logs found")
        return

    logger.info(f"Evaluating {len(logs)} log entries...")

    with get_db() as db:
        if not db:
            logger.error("Database connection failed")
            return

        for log in logs:
            log_id = log.get("id")
            if not log_id:
                continue

            question = log.get("user_prompt", "")
            answer_str = log.get("assistant_answer", "")
            messages = log.get("raw_json", {}).get("messages", [])

            searches, retrievals = count_tool_calls(messages)

            parsed = parse_answer(answer_str)

            follow_score = evaluate_follow_instruction(
                searches, retrievals, bool(parsed["answer"])
            )
            relevant_score = evaluate_answer_relevant(
                question, parsed["answer"] or "", parsed["sources"]
            )

            try:
                insert_eval_check(
                    db,
                    log_id=log_id,
                    check_name="follow_instruction",
                    passed=follow_score >= PASS_THRESHOLD,
                    score=follow_score,
                    details=None,
                )
                insert_eval_check(
                    db,
                    log_id=log_id,
                    check_name="answer_relevant",
                    passed=relevant_score >= PASS_THRESHOLD,
                    score=relevant_score,
                    details=None,
                )
                logger.info(
                    f"✅ Log {log_id}: follow={follow_score:.2f}, relevant={relevant_score:.2f}"
                )
            except Exception as e:
                logger.error(f"Error saving log {log_id}: {e}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
