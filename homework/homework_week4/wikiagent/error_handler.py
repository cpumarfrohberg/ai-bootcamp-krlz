import logging
from typing import Any

from wikiagent.config import ERROR_MAPPINGS, ErrorCategory
from wikiagent.models import AgentError, WikipediaAgentResponse

logger = logging.getLogger(__name__)


def handle_agent_error(
    exception: Exception,
    tool_calls: list[dict[str, Any]],
) -> WikipediaAgentResponse:
    """
    Convert exception to structured error response.

    Args:
        exception: The exception that occurred
        tool_calls: List of tool calls made before the error

    Returns:
        WikipediaAgentResponse with error information
    """
    logger.error(f"Error during agent execution: {exception}")
    error_type = type(exception).__name__
    error_msg = str(exception).lower()
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
            technical_details=str(exception),
        )
    else:
        agent_error = AgentError(
            error_type=error_type,
            message=f"An error occurred: {error_type}",
            suggestion="Please try again. If the problem persists, check your internet connection and API configuration.",
            technical_details=str(exception),
        )

    return WikipediaAgentResponse(
        answer=None,
        tool_calls=tool_calls,
        usage=None,
        error=agent_error,
    )
