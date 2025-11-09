"""Tests for LLM-as-a-Judge evaluation"""

import pytest

from evals.judge import evaluate_answer
from wikiagent.config import (
    MAX_SCORE,
    MIN_REASONING_LENGTH,
    MIN_SCORE,
    TEST_ANSWER,
    TEST_CONFIDENCE,
    TEST_QUESTION,
    TEST_REASONING,
    TEST_SOURCES,
)
from wikiagent.models import SearchAgentAnswer, TokenUsage


@pytest.fixture
def test_question():
    """Test question for judge evaluation"""
    return TEST_QUESTION


@pytest.fixture
def test_answer():
    """Test SearchAgentAnswer for judge evaluation"""
    return SearchAgentAnswer(
        answer=TEST_ANSWER,
        confidence=TEST_CONFIDENCE,
        sources_used=TEST_SOURCES,
        reasoning=TEST_REASONING,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_judge_evaluates_answer(test_question, test_answer):
    """Test that judge evaluates answer and returns structured output"""
    result = await evaluate_answer(test_question, test_answer)

    # Verify result structure
    assert hasattr(result, "evaluation")
    assert hasattr(result, "usage")

    # Verify evaluation structure
    evaluation = result.evaluation
    assert (
        evaluation.overall_score >= MIN_SCORE and evaluation.overall_score <= MAX_SCORE
    )
    assert evaluation.accuracy >= MIN_SCORE and evaluation.accuracy <= MAX_SCORE
    assert evaluation.completeness >= MIN_SCORE and evaluation.completeness <= MAX_SCORE
    assert evaluation.relevance >= MIN_SCORE and evaluation.relevance <= MAX_SCORE
    assert isinstance(evaluation.reasoning, str)
    assert len(evaluation.reasoning) >= MIN_REASONING_LENGTH

    # Verify usage information
    usage = result.usage
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens >= 0
    assert usage.output_tokens >= 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_judge_output_structure(test_question, test_answer):
    """Test that judge output has correct structure"""
    result = await evaluate_answer(test_question, test_answer)

    # Verify result structure
    assert hasattr(result, "evaluation")
    assert hasattr(result, "usage")

    # Verify all required fields are present in evaluation
    evaluation = result.evaluation
    assert hasattr(evaluation, "overall_score")
    assert hasattr(evaluation, "accuracy")
    assert hasattr(evaluation, "completeness")
    assert hasattr(evaluation, "relevance")
    assert hasattr(evaluation, "reasoning")

    # Verify usage structure
    usage = result.usage
    assert isinstance(usage, TokenUsage)
    assert hasattr(usage, "input_tokens")
    assert hasattr(usage, "output_tokens")
    assert hasattr(usage, "total_tokens")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_judge_scores_in_range(test_question, test_answer):
    """Test that all judge scores are in valid range (0.0 to 1.0)"""
    result = await evaluate_answer(test_question, test_answer)

    evaluation = result.evaluation
    assert MIN_SCORE <= evaluation.overall_score <= MAX_SCORE
    assert MIN_SCORE <= evaluation.accuracy <= MAX_SCORE
    assert MIN_SCORE <= evaluation.completeness <= MAX_SCORE
    assert MIN_SCORE <= evaluation.relevance <= MAX_SCORE
