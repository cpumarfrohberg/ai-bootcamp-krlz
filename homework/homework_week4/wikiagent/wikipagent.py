import json
import logging
from collections.abc import Coroutine
from typing import Any, Callable, List

from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent

from config import DEFAULT_SEARCH_MODE, OPENAI_RAG_MODEL, SearchMode
from wikiagent.config import MAX_QUESTION_LOG_LENGTH
from wikiagent.create_agent import create_wikipedia_agent
from wikiagent.error_handler import handle_agent_error
from wikiagent.models import TokenUsage, WikipediaAgentResponse
from wikiagent.tools import wikipedia_get_page, wikipedia_search

logger = logging.getLogger(__name__)

# Import logging function (lazy import to avoid circular dependencies)
try:
    from wikiagent.agent_logging import log_agent_run
except ImportError:
    log_agent_run = None
    logger.warning("agent_logging module not available, logging will be skipped")

# Import guardrail functions (lazy import to avoid circular dependencies)
try:
    from wikiagent.guardrails import check_query_guardrail
except ImportError:
    check_query_guardrail = None
    logger.warning("guardrails module not available, guardrails will be skipped")

QUERY_DISPLAY_LENGTH = 50
STREAM_DEBOUNCE = 0.01
STRUCTURED_OUTPUT_FIELDS = ["answer", "confidence", "sources_used", "reasoning"]


def _parse_tool_args(args: Any, max_length: int = QUERY_DISPLAY_LENGTH) -> str:
    """Extract query from tool args for display purposes"""
    try:
        args_dict = json.loads(args) if isinstance(args, str) else args
        query = (
            args_dict.get("query", "N/A")[:max_length]
            if isinstance(args_dict, dict)
            else str(args)[:max_length]
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        query = str(args)[:max_length] if args else "N/A"
    return query


def _create_tool_call_tracker(
    tool_calls: List[dict],
) -> Callable[[Any, Any], Coroutine[Any, Any, None]]:
    """Create a tool call tracker function that appends to the provided list"""

    async def track_tool_calls(ctx: Any, event: Any) -> None:
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await track_tool_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_call = {
                "tool_name": event.part.tool_name,
                "args": event.part.args,
            }
            tool_calls.append(tool_call)
            query = _parse_tool_args(event.part.args)
            logger.info(
                f"Tool Call #{len(tool_calls)}: {event.part.tool_name} with query: {query}..."
            )

    return track_tool_calls


def _extract_token_usage(usage_obj: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage_obj.input_tokens,
        output_tokens=usage_obj.output_tokens,
        total_tokens=usage_obj.input_tokens + usage_obj.output_tokens,
    )


def _is_structured_output(args_str: str) -> bool:
    """Check if args contain structured output fields"""
    return any(field in args_str.lower() for field in STRUCTURED_OUTPUT_FIELDS)


def _calculate_delta(current_text: str, previous_text: str) -> str:
    """Calculate delta between current and previous text"""
    return current_text[len(previous_text) :]


def _process_streaming_part(
    part: Any,
    tool_call_callback: Callable[[str, str], None] | None,
    structured_output_callback: Callable[[str], None] | None,
    previous_text: str,
) -> tuple[str, bool]:
    """
    Process a single streaming part.
    Returns: (updated_previous_text, handled)
    """
    if not hasattr(part, "tool_name"):
        return previous_text, False

    tool_name = part.tool_name
    args = part.args

    # Handle Wikipedia tool calls
    if tool_name in {"wikipedia_search", "wikipedia_get_page"}:
        if tool_call_callback:
            tool_call_callback(tool_name, args)
        return previous_text, True

    # Handle structured output
    if tool_name and args:
        args_str = args if isinstance(args, str) else json.dumps(args)
        if _is_structured_output(args_str):
            delta = _calculate_delta(args_str, previous_text)
            if structured_output_callback and delta:
                structured_output_callback(delta)
            return args_str, True

    return previous_text, False


async def query_wikipedia(
    question: str,
    openai_model: str = OPENAI_RAG_MODEL,
    search_mode: SearchMode = DEFAULT_SEARCH_MODE,
    agent: Agent | None = None,
) -> WikipediaAgentResponse:
    """Query Wikipedia using the agent with search and get_page tools."""
    tool_calls: List[dict] = []
    if agent is None:
        agent = create_wikipedia_agent(openai_model, search_mode)
    logger.info(
        f"Running Wikipedia agent query: {question[:MAX_QUESTION_LOG_LENGTH]}..."
    )

    try:
        track_handler = _create_tool_call_tracker(tool_calls)
        result = await agent.run(question, event_stream_handler=track_handler)
    except Exception as e:
        return handle_agent_error(e, tool_calls)

    logger.info(f"Agent completed query. Tool calls: {len(tool_calls)}")
    usage = _extract_token_usage(result.usage())
    logger.info(
        f"Token usage - Input: {usage.input_tokens}, Output: {usage.output_tokens}, Total: {usage.total_tokens}"
    )
    return WikipediaAgentResponse(
        answer=result.output,
        tool_calls=tool_calls,
        usage=usage,
    )


async def query_wikipedia_stream(
    question: str,
    openai_model: str = OPENAI_RAG_MODEL,
    search_mode: SearchMode = DEFAULT_SEARCH_MODE,
    tool_call_callback: Callable[[str, str], None] | None = None,
    structured_output_callback: Callable[[str], None] | None = None,
    agent: Agent | None = None,
) -> WikipediaAgentResponse:
    """Query Wikipedia using the agent with streaming support for real-time updates."""
    tool_calls: List[dict] = []
    if agent is None:
        agent = create_wikipedia_agent(openai_model, search_mode)
    logger.info(
        f"Running Wikipedia agent query with streaming: {question[:MAX_QUESTION_LOG_LENGTH]}..."
    )

    # Check query guardrail first
    if check_query_guardrail:
        from wikiagent.config import GUARDRAIL_BLOCKED_KEYWORDS

        guardrail_error = await check_query_guardrail(
            question, GUARDRAIL_BLOCKED_KEYWORDS
        )
        if guardrail_error:
            return guardrail_error

    previous_text = ""
    result: Any = None
    final_output: Any = None

    try:
        track_handler = _create_tool_call_tracker(tool_calls)
        async with agent.run_stream(
            question, event_stream_handler=track_handler
        ) as stream_result:
            result = stream_result
            async for item, last in result.stream_responses(
                debounce_by=STREAM_DEBOUNCE
            ):
                for part in item.parts:
                    previous_text, _ = _process_streaming_part(
                        part,
                        tool_call_callback,
                        structured_output_callback,
                        previous_text,
                    )
            final_output = await result.get_output()

        usage = _extract_token_usage(result.usage())
        logger.info(
            f"📊 Token Usage - Input: {usage.input_tokens:,}, Output: {usage.output_tokens:,}, Total: {usage.total_tokens:,}"
        )

        # Save log to database
        if log_agent_run:
            await log_agent_run(agent, result, question)

        return WikipediaAgentResponse(
            answer=final_output,
            tool_calls=tool_calls,
            usage=usage,
        )
    except Exception as e:
        return handle_agent_error(e, tool_calls)
