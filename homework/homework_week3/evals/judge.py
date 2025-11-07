"""LLM-as-a-Judge for evaluating Wikipedia agent answers"""

import logging

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


async def evaluate_answer(
    question: str,
    answer: RAGAnswer,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> JudgeEvaluation:
    """
    Evaluate answer quality using LLM-as-a-Judge.

    Args:
        question: The original question asked
        answer: The RAGAnswer from the Wikipedia agent
        judge_model: Model to use for judging (default: DEFAULT_JUDGE_MODEL)

    Returns:
        JudgeEvaluation with scores and reasoning
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

    # Prepare evaluation prompt
    evaluation_prompt = f"""Question: {question}

Answer: {answer.answer}

Sources used: {', '.join(answer.sources_used) if answer.sources_used else 'None'}

Evaluate this answer on accuracy, completeness, and relevance to the question."""

    logger.info(f"Evaluating answer for question: {question[:50]}...")
    print("⚖️  Judge is evaluating the answer...")

    # Run judge evaluation
    try:
        result = await judge_agent.run(evaluation_prompt)
        evaluation = result.output

        logger.info(
            f"Judge evaluation complete. Overall score: {evaluation.overall_score:.2f}"
        )
        print(
            f"✅ Judge evaluation complete. Overall score: {evaluation.overall_score:.2f}"
        )

        return evaluation

    except Exception as e:
        logger.error(f"Error during judge evaluation: {e}")
        raise
