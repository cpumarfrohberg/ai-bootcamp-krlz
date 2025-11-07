"""Tests for LLM-as-a-Judge evaluation"""

import pytest

from evals.judge import evaluate_answer
from wikiagent.models import RAGAnswer

TEST_QUESTION = "What factors influence customer behavior?"
TEST_ANSWER = "Customer behavior is influenced by multiple factors including psychological factors (motivation, perception), social factors (family, culture), personal factors (age, lifestyle), and marketing factors (product, price, promotion, place)."
TEST_CONFIDENCE = 0.95
TEST_SOURCES = ["Consumer behaviour", "Behavioral economics"]
TEST_REASONING = (
    "Found relevant Wikipedia pages on factors influencing customer behavior"
)

MIN_SCORE = 0.0
MAX_SCORE = 1.0
MIN_REASONING_LENGTH = 1


@pytest.fixture
def test_question():
    """Test question for judge evaluation"""
    return TEST_QUESTION


@pytest.fixture
def test_answer():
    """Test RAGAnswer for judge evaluation"""
    return RAGAnswer(
        answer=TEST_ANSWER,
        confidence=TEST_CONFIDENCE,
        sources_used=TEST_SOURCES,
        reasoning=TEST_REASONING,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_judge_evaluates_answer(test_question, test_answer):
    """Test that judge evaluates answer and returns structured output"""
    evaluation = await evaluate_answer(test_question, test_answer)

    # Verify evaluation structure
    assert (
        evaluation.overall_score >= MIN_SCORE and evaluation.overall_score <= MAX_SCORE
    )
    assert evaluation.accuracy >= MIN_SCORE and evaluation.accuracy <= MAX_SCORE
    assert evaluation.completeness >= MIN_SCORE and evaluation.completeness <= MAX_SCORE
    assert evaluation.relevance >= MIN_SCORE and evaluation.relevance <= MAX_SCORE
    assert isinstance(evaluation.reasoning, str)
    assert len(evaluation.reasoning) >= MIN_REASONING_LENGTH


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_judge_output_structure(test_question, test_answer):
    """Test that judge output has correct structure"""
    evaluation = await evaluate_answer(test_question, test_answer)

    # Verify all required fields are present
    assert hasattr(evaluation, "overall_score")
    assert hasattr(evaluation, "accuracy")
    assert hasattr(evaluation, "completeness")
    assert hasattr(evaluation, "relevance")
    assert hasattr(evaluation, "reasoning")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_judge_scores_in_range(test_question, test_answer):
    """Test that all judge scores are in valid range (0.0 to 1.0)"""
    evaluation = await evaluate_answer(test_question, test_answer)

    assert MIN_SCORE <= evaluation.overall_score <= MAX_SCORE
    assert MIN_SCORE <= evaluation.accuracy <= MAX_SCORE
    assert MIN_SCORE <= evaluation.completeness <= MAX_SCORE
    assert MIN_SCORE <= evaluation.relevance <= MAX_SCORE
