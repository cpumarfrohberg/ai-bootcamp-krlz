# Prompt building utilities
import json
from typing import Any, Dict, List

from config import (
    InstructionType,
    PromptTemplate,
)


def build_prompt(
    question: str,
    search_results: List[Dict[str, Any]],
    instruction_type: InstructionType = InstructionType.PODCAST_ASSISTANT,
) -> str:
    """
    Build the prompt for the LLM using the question and search results.

    Args:
        question: The user's question
        search_results: List of relevant document dictionaries from search
        instruction_type: Type of instruction/prompt template to use

    Returns:
        Formatted prompt string ready for LLM input
    """
    prompt_template: str = PromptTemplate[instruction_type.name].value
    search_json: str = json.dumps(search_results)
    return prompt_template.format(question=question, context=search_json)
