# Configuration for RAG Agent
"""RAG Agent configuration dataclass"""

from dataclasses import dataclass

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CONTEXT_LENGTH,
    MONGODB_DB,
    MONGODB_URI,
    OPENAI_RAG_MODEL,
    InstructionType,
    SearchType,
)

# User-Agent header required by Wikipedia API
USER_AGENT = "WikipediaAgent/1.0 (https://github.com/yourusername/wikipedia-agent)"

# Test constants for agent tests
TEST_QUESTIONS = [
    "What factors influence customer behavior?",
    "How do users behave on websites?",
    "What is customer satisfaction?",
    "What are user behavior patterns?",
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


@dataclass
class RAGConfig:
    """Configuration for RAG system"""

    search_type: SearchType = SearchType.SENTENCE_TRANSFORMERS
    openai_model: str = OPENAI_RAG_MODEL  # OpenAI model name (e.g., "gpt-4o-mini")
    instruction_type: InstructionType = InstructionType.WIKIPEDIA_AGENT
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH
    max_tool_calls: int = 3  # Maximum number of tool calls allowed (safety limit)
    mongo_uri: str = MONGODB_URI
    database: str = MONGODB_DB
    collection: str = "posts"
