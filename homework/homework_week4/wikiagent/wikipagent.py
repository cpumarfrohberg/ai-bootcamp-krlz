import json
import logging
from collections.abc import Coroutine
from typing import Any, Callable, List

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import DEFAULT_MAX_TOKENS, DEFAULT_SEARCH_MODE, OPENAI_RAG_MODEL, SearchMode
from config.adaptive_instructions import get_wikipedia_agent_instructions
from wikiagent.config import ERROR_MAPPINGS, MAX_QUESTION_LOG_LENGTH, ErrorCategory
from wikiagent.models import (
    AgentError,
    SearchAgentAnswer,
    TokenUsage,
    WikipediaAgentResponse,
)
from wikiagent.tools import wikipedia_get_page, wikipedia_search

logger = logging.getLogger(__name__)

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


def _create_agent(openai_model: str, search_mode: SearchMode) -> Agent:
    instructions = get_wikipedia_agent_instructions(search_mode)
    model = OpenAIChatModel(model_name=openai_model, provider=OpenAIProvider())
    logger.info(f"Using OpenAI model: {openai_model}, search mode: {search_mode}")
    return Agent(
        name="wikipedia_agent",
        model=model,
        tools=[wikipedia_search, wikipedia_get_page],
        instructions=instructions,
        output_type=SearchAgentAnswer,
        model_settings=ModelSettings(max_tokens=DEFAULT_MAX_TOKENS),
        end_strategy="exhaustive",
    )


def _handle_error(e: Exception, tool_calls: List[dict]) -> WikipediaAgentResponse:
    """Convert exception to structured error response"""
    logger.error(f"Error during agent execution: {e}")
    error_type = type(e).__name__
    error_msg = str(e).lower()
    error_type_lower = error_type.lower()

    # Find matching error category
    error_config = None
    for category in ErrorCategory:
        mapping = ERROR_MAPPINGS[category]
        keywords = mapping.get("keywords", [])
        matches_error_msg = any(kw in error_msg for kw in keywords)
        matches_error_type = any(kw in error_type_lower for kw in keywords)
        if matches_error_msg or matches_error_type:
            error_config = mapping
            break

    if error_config:
        agent_error = AgentError(
            error_type=error_config["error_type"],
            message=error_config["message"],
            suggestion=error_config["suggestion"],
            technical_details=str(e),
        )
    else:
        agent_error = AgentError(
            error_type=error_type,
            message=f"An error occurred: {error_type}",
            suggestion="Please try again. If the problem persists, check your internet connection and API configuration.",
            technical_details=str(e),
        )

    return WikipediaAgentResponse(
        answer=None,
        tool_calls=tool_calls,
        usage=None,
        error=agent_error,
    )


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
        agent = _create_agent(openai_model, search_mode)
    logger.info(
        f"Running Wikipedia agent query: {question[:MAX_QUESTION_LOG_LENGTH]}..."
    )

    try:
        track_handler = _create_tool_call_tracker(tool_calls)
        result = await agent.run(question, event_stream_handler=track_handler)
    except Exception as e:
        return _handle_error(e, tool_calls)

    logger.info(f"Agent completed query. Tool calls: {len(tool_calls)}")
    usage = _extract_token_usage(result.usage())
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
        agent = _create_agent(openai_model, search_mode)
    logger.info(
        f"Running Wikipedia agent query with streaming: {question[:MAX_QUESTION_LOG_LENGTH]}..."
    )
    previous_text = ""

    try:
        track_handler = _create_tool_call_tracker(tool_calls)
        async with agent.run_stream(
            question, event_stream_handler=track_handler
        ) as result:
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
            return WikipediaAgentResponse(
                answer=final_output,
                tool_calls=tool_calls,
                usage=usage,
            )
    except Exception as e:
        return _handle_error(e, tool_calls)
