"""Wikipedia Agent configuration constants"""

USER_AGENT = "WikipediaAgent/1.0 (https://github.com/yourusername/wikipedia-agent)"

# Maximum page content length to prevent context overflow
# Truncate Wikipedia page content to this length (characters)
# With ~4 chars per token, 15000 chars ≈ 3750 tokens per page
# With 15-20 pages, this keeps us under 128k token limit
MAX_PAGE_CONTENT_LENGTH = 15000  # characters

# Test constants for agent tests
# Using simpler, more direct questions for better testing
TEST_QUESTIONS = [
    "What is customer satisfaction?",
    "What is consumer behaviour?",
    "What is user experience?",
    "What is behavioral economics?",
]

# Minimum expected tool calls based on agent instructions
# Phase 1: 3-5 broad searches, Phase 2: 8-12 specific searches
# Total: ~11-17 searches expected
MIN_SEARCH_CALLS = 3  # At least 3 searches (minimum from Phase 1)
MIN_GET_PAGE_CALLS = 2  # At least 2 page retrievals (multiple pages expected)

# Test constants for judge tests
TEST_QUESTION = "What factors influence customer behavior?"
TEST_ANSWER = "Customer behavior is influenced by multiple factors including psychological factors (motivation, perception), social factors (family, culture), personal factors (age, lifestyle), and marketing factors (product, price, promotion, place)."
TEST_CONFIDENCE = 0.95
TEST_SOURCES = ["Consumer behaviour", "Behavioral economics"]
TEST_REASONING = (
    "Found relevant Wikipedia pages on factors influencing customer behavior"
)

# Judge evaluation score constants
MIN_SCORE = 0.0
MAX_SCORE = 1.0
MIN_REASONING_LENGTH = 1

# Logging constants
MAX_QUESTION_LOG_LENGTH = 100  # Max length for question in logs
