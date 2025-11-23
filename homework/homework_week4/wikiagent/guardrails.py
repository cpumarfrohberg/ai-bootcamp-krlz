import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_COST_CHECK_INTERVAL = 0.5  # seconds


@dataclass
class GuardrailFunctionOutput:
    output_info: str
    tripwire_triggered: bool


class GuardrailException(Exception):
    """Exception raised when a guardrail is triggered."""

    def __init__(self, message: str, info: GuardrailFunctionOutput):
        super().__init__(message)
        self.info = info


async def cost_guardrail(
    max_cost: float,
    cost_tracker: dict,
    check_interval: float = DEFAULT_COST_CHECK_INTERVAL,
) -> None:
    """
    Monitor cost during execution.

    Args:
        max_cost: Maximum allowed cost
        cost_tracker: Shared dict with 'total' key (updated by agent)
        check_interval: How often to check (seconds)

    Raises:
        GuardrailException if cost exceeds max_cost
    """
    while True:
        await asyncio.sleep(check_interval)
        current_cost = cost_tracker.get("total", 0.0)
        if current_cost > max_cost:
            raise GuardrailException(
                f"Cost limit exceeded: ${current_cost:.2f} (limit: ${max_cost:.2f})",
                GuardrailFunctionOutput(
                    output_info=f"Cost limit {max_cost} exceeded",
                    tripwire_triggered=True,
                ),
            )


async def query_guardrail(
    question: str,
    blocked_keywords: list[str],
) -> None:
    """
    Check query for inappropriate content.

    Args:
        question: User's question
        blocked_keywords: List of keywords to block

    Raises:
        GuardrailException if blocked keyword found
    """
    if not blocked_keywords:
        return

    question_lower = question.lower()
    for keyword in blocked_keywords:
        if keyword and keyword.lower() in question_lower:
            raise GuardrailException(
                f"Query contains blocked keyword: {keyword}",
                GuardrailFunctionOutput(
                    output_info=f"Blocked keyword detected: {keyword}",
                    tripwire_triggered=True,
                ),
            )


async def run_with_guardrails(
    agent_coroutine: any,
    guardrails: list[any],
) -> any:
    """
    Run agent with parallel guardrails.

    Args:
        agent_coroutine: The agent execution coroutine
        guardrails: List of guardrail coroutines

    Returns:
        Agent result if successful

    Raises:
        GuardrailException if any guardrail triggers
    """
    agent_task = asyncio.create_task(agent_coroutine)
    guardrail_tasks = [asyncio.create_task(g) for g in guardrails]

    try:
        await asyncio.gather(agent_task, *guardrail_tasks)
        return agent_task.result()

    except GuardrailException as e:
        logger.warning(f"[guardrail fired] {e.info.output_info}")

        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            logger.info("[run_with_guardrails] agent cancelled")

        for t in guardrail_tasks:
            t.cancel()
        await asyncio.gather(*guardrail_tasks, return_exceptions=True)

        raise
