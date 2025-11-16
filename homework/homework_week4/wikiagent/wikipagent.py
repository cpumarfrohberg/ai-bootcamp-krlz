"""Wikipedia agent for answering questions using Wikipedia content"""

import json
import logging
from typing import Any, Callable, List

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import DEFAULT_MAX_TOKENS, DEFAULT_SEARCH_MODE, OPENAI_RAG_MODEL, SearchMode
from config.adaptive_instructions import get_wikipedia_agent_instructions
from wikiagent.config import MAX_QUESTION_LOG_LENGTH
from wikiagent.models import (
    AgentError,
    SearchAgentAnswer,
    TokenUsage,
    WikipediaAgentResponse,
)
from wikiagent.tools import wikipedia_get_page, wikipedia_search

logger = logging.getLogger(__name__)

# Constants
QUERY_DISPLAY_LENGTH = 50
STREAM_DEBOUNCE = 0.01

_tool_calls: List[dict] = []


async def track_tool_calls(ctx: Any, event: Any) -> None:
    """Event handler to track all tool calls"""
    global _tool_calls

    # Handle nested async streams
    if hasattr(event, "__aiter__"):
        async for sub in event:
            await track_tool_calls(ctx, sub)
        return

    # Track function tool calls
    if isinstance(event, FunctionToolCallEvent):
        tool_call = {
            "tool_name": event.part.tool_name,
            "args": event.part.args,
        }
        _tool_calls.append(tool_call)
        tool_num = len(_tool_calls)

        # Parse args to extract query for display
        try:
            args_dict = (
                json.loads(event.part.args)
                if isinstance(event.part.args, str)
                else event.part.args
            )
            query = (
                args_dict.get("query", "N/A")[:QUERY_DISPLAY_LENGTH]
                if isinstance(args_dict, dict)
                else str(event.part.args)[:QUERY_DISPLAY_LENGTH]
            )
        except (json.JSONDecodeError, AttributeError, TypeError):
            query = (
                str(event.part.args)[:QUERY_DISPLAY_LENGTH]
                if event.part.args
                else "N/A"
            )

        print(
            f"🔍 Tool call #{tool_num}: {event.part.tool_name} with query: {query}..."
        )
        logger.info(
            f"Tool Call #{tool_num}: {event.part.tool_name} with args: {event.part.args}"
        )


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


def _handle_error(e: Exception) -> WikipediaAgentResponse:
    """Convert exception to structured error response"""
    logger.error(f"Error during agent execution: {e}")
    error_type = type(e).__name__
    error_msg = str(e)

    if "Wikipedia" in error_msg or "HTTP" in error_msg:
        agent_error = AgentError(
            error_type="WikipediaAPI",
            message="Wikipedia API error. The page may not exist or the service is temporarily unavailable.",
            suggestion="Try rephrasing your question or asking about a different topic.",
            technical_details=error_msg,
        )
    elif "Connection" in error_type or "connection" in error_msg.lower():
        agent_error = AgentError(
            error_type="Network",
            message="Connection error. Please check your internet connection.",
            suggestion="The Wikipedia API could not be reached. Please try again in a moment.",
            technical_details=error_msg,
        )
    elif "Timeout" in error_type or "timeout" in error_msg.lower():
        agent_error = AgentError(
            error_type="Timeout",
            message="Request timed out. The Wikipedia API took too long to respond.",
            suggestion="Please try again with a simpler question or check your connection.",
            technical_details=error_msg,
        )
    else:
        agent_error = AgentError(
            error_type=error_type,
            message=f"An error occurred: {error_type}",
            suggestion="Please try again. If the problem persists, check your internet connection and API configuration.",
            technical_details=error_msg,
        )

    return WikipediaAgentResponse(
        answer=None,
        tool_calls=_tool_calls,
        usage=None,
        error=agent_error,
    )


def _extract_token_usage(usage_obj: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage_obj.input_tokens,
        output_tokens=usage_obj.output_tokens,
        total_tokens=usage_obj.input_tokens + usage_obj.output_tokens,
    )


async def query_wikipedia(
    question: str,
    openai_model: str = OPENAI_RAG_MODEL,
    search_mode: SearchMode = DEFAULT_SEARCH_MODE,
) -> WikipediaAgentResponse:
    """
    Query Wikipedia using the agent with search and get_page tools.

    The agent will:
    1. Use wikipedia_search to find relevant pages
    2. Use wikipedia_get_page to retrieve full content
    3. Answer the question based on retrieved content

    Args:
        question: User question to answer
        openai_model: OpenAI model name (default: from config)
        search_mode: Search mode (EVALUATION, PRODUCTION, or RESEARCH)
                     Default: EVALUATION (strict minimums for consistent testing)

    Returns:
        WikipediaAgentResponse with answer and tool calls
    """
    global _tool_calls
    _tool_calls = []
    agent = _create_agent(openai_model, search_mode)
    logger.info(
        f"Running Wikipedia agent query: {question[:MAX_QUESTION_LOG_LENGTH]}..."
    )
    print("🤖 Wikipedia Agent is processing your question...")

    try:
        result = await agent.run(question, event_stream_handler=track_tool_calls)
    except Exception as e:
        return _handle_error(e)

    logger.info(f"Agent completed query. Tool calls: {len(_tool_calls)}")
    print(f"✅ Agent completed query. Made {len(_tool_calls)} tool calls.")
    usage = _extract_token_usage(result.usage())
    return WikipediaAgentResponse(
        answer=result.output,
        tool_calls=_tool_calls,
        usage=usage,
    )


async def query_wikipedia_stream(
    question: str,
    openai_model: str = OPENAI_RAG_MODEL,
    search_mode: SearchMode = DEFAULT_SEARCH_MODE,
    tool_call_callback: Callable[[str, str], None] | None = None,
    structured_output_callback: Callable[[str], None] | None = None,
) -> WikipediaAgentResponse:
    """
    Query Wikipedia using the agent with streaming support.

    This function streams the agent's response, allowing real-time updates
    of tool calls and structured output as they arrive.

    Args:
        question: User question to answer
        openai_model: OpenAI model name (default: from config)
        search_mode: Search mode (EVALUATION, PRODUCTION, or RESEARCH)
        tool_call_callback: Optional callback for tool calls (tool_name, args)
        structured_output_callback: Optional callback for structured output JSON deltas

    Returns:
        WikipediaAgentResponse with answer and tool calls
    """
    global _tool_calls
    _tool_calls = []
    agent = _create_agent(openai_model, search_mode)
    logger.info(
        f"Running Wikipedia agent query with streaming: {question[:MAX_QUESTION_LOG_LENGTH]}..."
    )
    previous_text = ""

    try:
        async with agent.run_stream(
            question, event_stream_handler=track_tool_calls
        ) as result:
            async for item, last in result.stream_responses(
                debounce_by=STREAM_DEBOUNCE
            ):
                for part in item.parts:
                    if not hasattr(part, "tool_name"):
                        continue

                    tool_name = part.tool_name
                    args = part.args

                    if tool_name in {"wikipedia_search", "wikipedia_get_page"}:
                        if tool_call_callback:
                            tool_call_callback(tool_name, args)
                        continue

                    if tool_name and args:
                        args_str = args if isinstance(args, str) else json.dumps(args)
                        is_structured_output = any(
                            field in args_str.lower()
                            for field in [
                                "answer",
                                "confidence",
                                "sources_used",
                                "reasoning",
                            ]
                        )

                        if is_structured_output:
                            delta = args_str[len(previous_text) :]
                            previous_text = args_str
                            if structured_output_callback and delta:
                                structured_output_callback(delta)

            final_output = await result.get_output()
            usage = _extract_token_usage(result.usage())
            return WikipediaAgentResponse(
                answer=final_output,
                tool_calls=_tool_calls,
                usage=usage,
            )
    except Exception as e:
        return _handle_error(e)
