"""LLM-as-a-Judge for evaluating Wikipedia agent answers"""

import asyncio
import json
import logging
from typing import Optional

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
from config.instructions import InstructionsConfig, InstructionType
from wikiagent.models import JudgeEvaluation, RAGAnswer

logger = logging.getLogger(__name__)


async def _run_judge_with_retry(
    judge_agent: Agent,
    prompt: str,
    max_retries: int = 3,
) -> tuple[JudgeEvaluation, dict]:
    """
    Run judge with exponential backoff retry logic.

    Args:
        judge_agent: The judge agent instance
        prompt: The evaluation prompt
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Tuple of (JudgeEvaluation, usage_dict) where usage_dict contains:
        - input_tokens: Number of input tokens used
        - output_tokens: Number of output tokens used
        - total_tokens: Total tokens used
    """
    for attempt in range(max_retries):
        try:
            result = await judge_agent.run(prompt)
            evaluation = result.output
            usage = result.usage()  # Get token usage

            return evaluation, {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed - return fallback
                logger.error(
                    f"Judge evaluation failed after {max_retries} attempts: {e}"
                )
                fallback_eval = JudgeEvaluation(
                    overall_score=0.0,
                    accuracy=0.0,
                    completeness=0.0,
                    relevance=0.0,
                    reasoning=f"Judge evaluation failed: {str(e)}",
                )
                return fallback_eval, {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2**attempt
            logger.warning(
                f"Judge evaluation attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
            )
            await asyncio.sleep(wait_time)

    # Should never reach here, but just in case
    raise RuntimeError("Judge evaluation failed after all retries")


async def evaluate_answer(
    question: str,
    answer: RAGAnswer,
    tool_calls: Optional[list[dict]] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = 3,
) -> tuple[JudgeEvaluation, dict]:
    """
    Evaluate answer quality using LLM-as-a-Judge.

    Args:
        question: The original question asked
        answer: The RAGAnswer from the Wikipedia agent
        tool_calls: Optional list of tool calls made by the agent (for context)
        judge_model: Model to use for judging (default: DEFAULT_JUDGE_MODEL)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Tuple of (JudgeEvaluation, usage_dict) where usage_dict contains:
        - input_tokens: Number of input tokens used
        - output_tokens: Number of output tokens used
        - total_tokens: Total tokens used
    """
    # Get judge instructions
    instructions = InstructionsConfig.INSTRUCTIONS[InstructionType.JUDGE]

    # Initialize judge model
    model = OpenAIChatModel(
        model_name=judge_model,
        provider=OpenAIProvider(),
    )
    logger.info(f"Using judge model: {judge_model}")

    # Create judge agent
    judge_agent = Agent(
        name="judge",
        model=model,
        instructions=instructions,
        output_type=JudgeEvaluation,
        model_settings=ModelSettings(
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_JUDGE_TEMPERATURE,
        ),
    )

    # Build tool calls section
    tool_calls_section = ""
    if tool_calls:
        tool_calls_section = f"""
<TOOL_CALLS>
{json.dumps(tool_calls, indent=2)}
</TOOL_CALLS>"""

    # Prepare evaluation prompt with XML tags (best practice from Evidently AI)
    evaluation_prompt = f"""<QUESTION>{question}</QUESTION>

<ANSWER>{answer.answer}</ANSWER>

<SOURCES>{', '.join(answer.sources_used) if answer.sources_used else 'None'}</SOURCES>{tool_calls_section}

Evaluate this answer on accuracy, completeness, and relevance to the question."""

    logger.info(f"Evaluating answer for question: {question[:50]}...")
    print("⚖️  Judge is evaluating the answer...")

    # Run judge evaluation with retry logic
    evaluation, usage = await _run_judge_with_retry(
        judge_agent, evaluation_prompt, max_retries
    )

    logger.info(
        f"Judge evaluation complete. Overall score: {evaluation.overall_score:.2f}, "
        f"Tokens: {usage['total_tokens']}"
    )
    print(
        f"✅ Judge evaluation complete. Overall score: {evaluation.overall_score:.2f}"
    )

    return evaluation, usage
