import json
import logging
from dataclasses import dataclass
from typing import Any

import pydantic
from genai_prices import Usage, calc_price
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)

UsageTypeAdapter = pydantic.TypeAdapter(RunUsage)


@dataclass
class CostResult:
    """Result of cost calculation."""

    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    error: str | None = None


def _create_log_entry(
    agent: Agent,
    messages: list[ModelMessage],
    usage: RunUsage,
    output: Any,
) -> dict:
    """
    Extract log data from agent execution.

    Args:
        agent: The agent instance
        messages: List of model messages from the run
        usage: Token usage information
        output: The agent output

    Returns:
        Dictionary containing log entry data
    """
    tools = []
    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)
    dict_usage = UsageTypeAdapter.dump_python(usage)

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": agent.model.system,
        "model": agent.model.model_name,
        "tools": tools,
        "messages": dict_messages,
        "usage": dict_usage,
        "output": output,
    }


async def _log_agent_run(
    agent: Agent,
    result: StreamedRunResult,
) -> dict:
    """
    Unified logging function for streaming runs.

    Args:
        agent: The agent instance
        result: Streamed run result

    Returns:
        Dictionary containing log entry data
    """
    output = await result.get_output()
    usage = result.usage()
    messages = result.all_messages()

    return _create_log_entry(
        agent=agent,
        messages=messages,
        usage=usage,
        output=output,
    )


def _calc_cost(
    provider: str | None,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> CostResult:
    """
    Calculate cost using genai_prices library.

    Args:
        provider: Provider name (e.g., 'openai')
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        CostResult with costs and optional error message
    """
    if not provider:
        error_msg = "Cost calculation skipped: provider not provided"
        logger.warning(error_msg)
        return CostResult(None, None, None, error=error_msg)

    if not model:
        error_msg = "Cost calculation skipped: model not provided"
        logger.warning(error_msg)
        return CostResult(None, None, None, error=error_msg)

    try:
        it = int(input_tokens or 0)
        ot = int(output_tokens or 0)
        token_usage = Usage(input_tokens=it, output_tokens=ot)
        price_data = calc_price(
            token_usage,
            provider_id=provider,
            model_ref=model,
        )
        return CostResult(
            input_cost=float(price_data.input_price),
            output_cost=float(price_data.output_price),
            total_cost=float(price_data.total_price),
        )
    except Exception as e:
        error_msg = f"Cost calculation failed: {e}"
        logger.warning(error_msg)
        return CostResult(None, None, None, error=error_msg)


async def save_log_to_db(
    agent: Agent,
    result: StreamedRunResult,
    question: str,
) -> int | None:
    """
    Save agent run to database.

    Args:
        agent: The agent instance
        result: Streamed run result
        question: The user's question

    Returns:
        Log ID if successful, None otherwise
    """
    try:
        from wikiagent.monitoring.db import get_db, insert_log
        from wikiagent.monitoring.schemas import LogCreate

        log_entry = await _log_agent_run(agent, result)
        usage = result.usage()

        # Extract answer from output
        answer_text = None
        if isinstance(log_entry["output"], dict):
            answer_text = log_entry["output"].get("answer", str(log_entry["output"]))
        elif log_entry["output"]:
            answer_text = str(log_entry["output"])

        cost_result = _calc_cost(
            log_entry["provider"],
            log_entry["model"],
            usage.input_tokens,
            usage.output_tokens,
        )

        # Convert system_prompt to string if it's a list
        instructions_text = log_entry["system_prompt"]
        if isinstance(instructions_text, list):
            instructions_text = "\n".join(str(item) for item in instructions_text)
        elif instructions_text is not None:
            instructions_text = str(instructions_text)

        # Validate data before database operation
        log_data = LogCreate(
            agent_name=log_entry["agent_name"],
            provider=log_entry["provider"],
            model=log_entry["model"],
            user_prompt=question,
            instructions=instructions_text,
            total_input_tokens=usage.input_tokens,
            total_output_tokens=usage.output_tokens,
            assistant_answer=answer_text,
            raw_json=json.dumps(log_entry, default=str),
            input_cost=cost_result.input_cost,
            output_cost=cost_result.output_cost,
            total_cost=cost_result.total_cost,
        )

        with get_db() as db:
            if not db:
                logger.warning("Database not available, skipping log save")
                return None

            log_id = insert_log(db, **log_data.model_dump())

        logger.info(f"Saved log to database with ID: {log_id}")
        return log_id

    except Exception as e:
        logger.error(f"Failed to save log to database: {e}", exc_info=True)
        return None


async def log_agent_run(
    agent: Agent,
    result: StreamedRunResult,
    question: str,
) -> int | None:
    """
    Log agent run to database with error handling.

    This is a public wrapper around save_log_to_db that provides
    consistent error handling and logging.

    Args:
        agent: The agent instance
        result: Streamed run result
        question: The user's question

    Returns:
        Log ID if successful, None otherwise
    """
    try:
        log_id = await save_log_to_db(agent, result, question)
        if log_id:
            logger.info(f"✅ Log saved to database with ID: {log_id}")
        else:
            logger.warning("⚠️ Failed to save log to database (log_id is None)")
        return log_id
    except Exception as e:
        logger.error(f"❌ Error saving log to database: {e}", exc_info=True)
        return None
