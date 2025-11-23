import logging

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import DEFAULT_MAX_TOKENS, SearchMode
from config.adaptive_instructions import get_wikipedia_agent_instructions
from wikiagent.models import SearchAgentAnswer
from wikiagent.tools import wikipedia_get_page, wikipedia_search

logger = logging.getLogger(__name__)


def create_wikipedia_agent(
    openai_model: str,
    search_mode: SearchMode,
) -> Agent:
    """
    Create a Wikipedia agent with the specified model and search mode.

    Args:
        openai_model: OpenAI model name (e.g., "gpt-4")
        search_mode: Search mode (evaluation, production, or research)

    Returns:
        Configured Agent instance
    """
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
