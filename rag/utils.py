# RAG and LLM utilities for question answering
import json
from typing import Any, Dict, List

from openai import OpenAI

from config import (
    DEFAULT_INSTRUCTION_TYPE,
    DEFAULT_MODEL,
    InstructionsConfig,
    InstructionType,
    PromptTemplate,
)


def _search_documents(question: str, index: Any | None = None) -> List[Dict[str, Any]]:
    """Search for relevant documents using the provided index."""
    from config import DEFAULT_BOOST_DICT, DEFAULT_FILTER_DICT, DEFAULT_NUM_RESULTS

    return index.search(
        question,
        boost_dict=DEFAULT_BOOST_DICT,
        filter_dict=DEFAULT_FILTER_DICT,
        num_results=DEFAULT_NUM_RESULTS,
    )


def _build_prompt(
    question: str,
    search_results: List[Dict[str, Any]],
    instruction_type: InstructionType = DEFAULT_INSTRUCTION_TYPE,
) -> str:
    """Build the prompt for the LLM using the question and search results."""
    prompt_template: str = PromptTemplate[instruction_type.name].value
    search_json: str = json.dumps(search_results)
    return prompt_template.format(question=question, context=search_json)


def _generate_response(
    user_prompt: str,
    instructions: str | None = None,
    model: str = DEFAULT_MODEL,
    openai_client: OpenAI | None = None,
) -> str:
    """Generate a response using the OpenAI API."""
    messages: List[Dict[str, str]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    messages.append({"role": "user", "content": user_prompt})

    response = openai_client.responses.create(model=model, input=messages)

    return response.output_text


def answer_question_with_context(
    question: str,
    index: Any | None = None,
    openai_client: OpenAI | None = None,
    instruction_type: InstructionType = DEFAULT_INSTRUCTION_TYPE,
) -> str:
    """
    Answer a question using RAG (Retrieval-Augmented Generation).

    Args:
        question: The question to answer
        index: The search index to retrieve relevant documents
        openai_client: OpenAI client for generating responses
        instruction_type: Type of instruction/prompt template to use

    Returns:
        Generated answer based on retrieved context
    """
    search_results: List[Dict[str, Any]] = _search_documents(question, index)
    user_prompt: str = _build_prompt(question, search_results, instruction_type)
    instructions: str = InstructionsConfig.INSTRUCTIONS[instruction_type]
    return _generate_response(
        user_prompt, instructions=instructions, openai_client=openai_client
    )
