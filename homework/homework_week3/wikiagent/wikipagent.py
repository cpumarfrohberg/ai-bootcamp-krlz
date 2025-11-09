"""Wikipedia agent for answering questions using Wikipedia content"""

import json
import logging
from typing import Any, List

from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent

from config import DEFAULT_MAX_TOKENS, DEFAULT_SEARCH_MODE, OPENAI_RAG_MODEL, SearchMode
from config.adaptive_instructions import get_wikipedia_agent_instructions
from config.instructions import InstructionsConfig, InstructionType
from wikiagent.models import SearchAgentAnswer, WikipediaAgentResponse

logger = logging.getLogger(__name__)

# Store tool calls for evaluation
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
                args_dict.get("query", "N/A")[:50]
                if isinstance(args_dict, dict)
                else str(event.part.args)[:50]
            )
        except (json.JSONDecodeError, AttributeError, TypeError):
            query = str(event.part.args)[:50] if event.part.args else "N/A"

        print(
            f"🔍 Tool call #{tool_num}: {event.part.tool_name} with query: {query}..."
        )
        logger.info(
            f"Tool Call #{tool_num}: {event.part.tool_name} with args: {event.part.args}"
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
    # Reset tool calls for this query
    global _tool_calls
    _tool_calls = []

    # Get instructions for Wikipedia agent (adaptive based on search_mode)
    instructions = get_wikipedia_agent_instructions(search_mode)

    # Initialize OpenAI model
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        model_name=openai_model,
        provider=OpenAIProvider(),
    )
    logger.info(f"Using OpenAI model: {openai_model}, search mode: {search_mode}")

    # Create agent with Wikipedia tools
    from pydantic_ai import ModelSettings

    from wikiagent.tools import wikipedia_get_page, wikipedia_search

    agent = Agent(
        name="wikipedia_agent",
        model=model,
        tools=[wikipedia_search, wikipedia_get_page],
        instructions=instructions,
        output_type=SearchAgentAnswer,
        model_settings=ModelSettings(max_tokens=DEFAULT_MAX_TOKENS),
        end_strategy="exhaustive",
    )

    logger.info(f"Running Wikipedia agent query: {question[:100]}...")
    print("🤖 Wikipedia Agent is processing your question...")

    # Run agent with event tracking
    try:
        result = await agent.run(
            question,
            event_stream_handler=track_tool_calls,
        )
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        raise

    logger.info(f"Agent completed query. Tool calls: {len(_tool_calls)}")
    print(f"✅ Agent completed query. Made {len(_tool_calls)} tool calls.")

    # Get token usage from agent result
    usage_obj = result.usage()
    from wikiagent.models import TokenUsage

    usage = TokenUsage(
        input_tokens=usage_obj.input_tokens,
        output_tokens=usage_obj.output_tokens,
        total_tokens=usage_obj.input_tokens + usage_obj.output_tokens,
    )

    return WikipediaAgentResponse(
        answer=result.output,
        tool_calls=_tool_calls,
        usage=usage,
    )
