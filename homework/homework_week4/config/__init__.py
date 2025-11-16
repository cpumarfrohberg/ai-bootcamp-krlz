import os
from enum import StrEnum

from dotenv import load_dotenv

from config.instructions import InstructionsConfig, InstructionType

load_dotenv(override=True)


class TokenizerModel(StrEnum):
    """Tokenizer models for token counting"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4O = "gpt-4o"


class TokenizerEncoding(StrEnum):
    """Tokenizer encoding fallbacks"""

    CL100K_BASE = "cl100k_base"
    P50K_BASE = "p50k_base"
    R50K_BASE = "r50k_base"


class SearchMode(StrEnum):
    """Search strategy modes for Wikipedia agent"""

    EVALUATION = (
        "evaluation"  # Strict minimums, no early stopping (for consistent testing)
    )
    PRODUCTION = (
        "production"  # Adaptive early stopping after minimum (for user queries)
    )
    RESEARCH = (
        "research"  # Maximum searches, no early stopping (for comprehensive research)
    )


DEFAULT_RAG_MODEL = TokenizerModel.GPT_4O_MINI
DEFAULT_JUDGE_MODEL = TokenizerModel.GPT_4O

DEFAULT_TEMPERATURE = 0.3  # Lower temperature for more focused, deterministic responses
DEFAULT_RAG_TEMPERATURE = 0.3
DEFAULT_JUDGE_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000  # Increased for quantized models that need more tokens for JSON output - relevant for ollama
DEFAULT_SCORE_ALPHA = 2.0
DEFAULT_SCORE_BETA = 0.5
DEFAULT_SCORE_GAMMA = 1.5
DEFAULT_TOKEN_NORMALIZATION_DIVISOR = 1000.0

# Search strategy configuration
DEFAULT_SEARCH_MODE = SearchMode.EVALUATION
MIN_SEARCH_CALLS = 3  # Minimum searches for consistency
MAX_SEARCH_CALLS = 8  # Maximum searches for cost control (production mode)
EARLY_STOP_CONFIDENCE_THRESHOLD = 0.9  # Confidence threshold for early stopping

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_RAG_MODEL = os.getenv("OPENAI_RAG_MODEL", str(DEFAULT_RAG_MODEL))
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", str(DEFAULT_JUDGE_MODEL))


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
