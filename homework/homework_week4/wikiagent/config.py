import os
from enum import StrEnum

USER_AGENT = "WikipediaAgent/1.0 (https://github.com/yourusername/wikipedia-agent)"

DATABASE_URL = os.getenv("DATABASE_URL")

# Guardrail configuration
GUARDRAIL_MAX_COST = float(os.getenv("GUARDRAIL_MAX_COST", "1.0"))
GUARDRAIL_BLOCKED_KEYWORDS = [
    kw.strip()
    for kw in os.getenv("GUARDRAIL_BLOCKED_KEYWORDS", "").split(",")
    if kw.strip()
]


class ErrorCategory(StrEnum):
    """Error categories for agent error handling"""

    WIKIPEDIA = "wikipedia"
    CONNECTION = "connection"
    TIMEOUT = "timeout"


# Maximum page content length to prevent context overflow
# Truncate Wikipedia page content to this length (characters)
# With ~4 chars per token, 15000 chars ≈ 3750 tokens per page
# With 15-20 pages, this keeps us under 128k token limit
MAX_PAGE_CONTENT_LENGTH = 15000  # characters

# Test constants for agent tests
# Using simpler, more direct questions for better testing
TEST_QUESTIONS = [
    "Where do capybaras live?",
    "What do capybaras eat?",
    "How big are capybaras?",
    "What is the habitat of capybaras?",
]

# Minimum expected tool calls based on agent instructions
# Phase 1: 3-5 broad searches, Phase 2: 8-12 specific searches
# Total: ~11-17 searches expected
MIN_SEARCH_CALLS = 3  # At least 3 searches (minimum from Phase 1)
MIN_GET_PAGE_CALLS = 2  # At least 2 page retrievals (multiple pages expected)

# Test constants for judge tests
TEST_QUESTION = "Where do capybaras live?"
TEST_ANSWER = "Capybaras live in South America, primarily in countries like Brazil, Venezuela, Colombia, and Argentina. They inhabit areas near water such as rivers, lakes, swamps, and marshes, as they are semi-aquatic animals that need water for both drinking and protection from predators."
TEST_CONFIDENCE = 0.95
TEST_SOURCES = ["Capybara", "South America", "Rodent"]
TEST_REASONING = "Found relevant Wikipedia pages on capybara habitat and distribution"

# Judge evaluation score constants
MIN_SCORE = 0.0
MAX_SCORE = 1.0
MIN_REASONING_LENGTH = 1

# Logging constants
MAX_QUESTION_LOG_LENGTH = 100  # Max length for question in logs

# Validation constants
MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 300
MIN_TITLE_LENGTH = 1
MAX_TITLE_LENGTH = 255

# Error mappings for agent error handling
ERROR_MAPPINGS = {
    ErrorCategory.WIKIPEDIA: {
        "error_type": "WikipediaAPI",
        "message": "Wikipedia API error. The page may not exist or the service is temporarily unavailable.",
        "suggestion": "Try rephrasing your question or asking about a different topic.",
        "keywords": ["wikipedia", "http"],
    },
    ErrorCategory.CONNECTION: {
        "error_type": "Network",
        "message": "Connection error. Please check your internet connection.",
        "suggestion": "The Wikipedia API could not be reached. Please try again in a moment.",
        "keywords": ["connection"],
    },
    ErrorCategory.TIMEOUT: {
        "error_type": "Timeout",
        "message": "Request timed out. The Wikipedia API took too long to respond.",
        "suggestion": "Please try again with a simpler question or check your connection.",
        "keywords": ["timeout"],
    },
}
