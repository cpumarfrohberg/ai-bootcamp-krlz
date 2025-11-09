import os
from enum import StrEnum

from dotenv import load_dotenv

# Import from instructions module (must be after StrEnum import)
from config.instructions import InstructionsConfig, InstructionType

load_dotenv(override=True)  # Override environment variables with .env values


class ModelType(StrEnum):
    """OpenAI model types for LLM inference"""

    GPT_4O_MINI = (
        "gpt-4o-mini"  # Default: Good balance of cost and quality, supports tools
    )
    GPT_3_5_TURBO = "gpt-3.5-turbo"  # Faster and cheaper, good for simple queries
    GPT_4O = "gpt-4o"  # Higher quality, better reasoning, more expensive
    GPT_4 = "gpt-4"  # Legacy GPT-4 model


class SearchType(StrEnum):
    MINSEARCH = "minsearch"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class SentenceTransformerModel(StrEnum):
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"


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


class DataType(StrEnum):
    """Types of data stored in MongoDB"""

    RAW_QUESTIONS = "raw_questions"
    RAW_ANSWERS = "raw_answers"
    PARSED_DOCUMENTS = "parsed_documents"
    RAG_INDEX = "rag_index"


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
DEFAULT_JUDGE_MODEL = TokenizerModel.GPT_4O  # Using gpt-4o for Judge (more capable than agent model, avoids self-consistency bias)
DEFAULT_SEARCH_TYPE = (
    SearchType.MINSEARCH
)  # Better performance, faster, and more efficient per evaluation results
DEFAULT_SENTENCE_TRANSFORMER_MODEL = SentenceTransformerModel.ALL_MINILM_L6_V2
DEFAULT_CHUNK_SIZE = (
    500  # Fewer chunks for speed (score: 1.157, tokens: 747.66, still perfect accuracy)
)
DEFAULT_CHUNK_OVERLAP = 0  # No overlap = faster chunking, fewer chunks
DEFAULT_CONTENT_FIELD = "content"
DEFAULT_MAX_CONTEXT_LENGTH = 800  # Reduced for speed (was 1000)

# LLM generation parameters
DEFAULT_TEMPERATURE = 0.3  # Lower temperature for more focused, deterministic responses
DEFAULT_RAG_TEMPERATURE = 0.3  # Temperature for RAG Agent (focused answers)
DEFAULT_JUDGE_TEMPERATURE = 0.1  # Lower temperature for Judge (consistent validation)
DEFAULT_MAX_TOKENS = (
    1000  # Increased for quantized models that need more tokens for JSON output
)

DEFAULT_NUM_RESULTS = 1

# Ground truth generation defaults
DEFAULT_GROUND_TRUTH_SAMPLES = int(os.getenv("DEFAULT_GROUND_TRUTH_SAMPLES", "50"))
DEFAULT_GROUND_TRUTH_OUTPUT = os.getenv(
    "DEFAULT_GROUND_TRUTH_OUTPUT", "evals/ground_truth.json"
)
DEFAULT_GROUND_TRUTH_MIN_TITLE_LENGTH = int(
    os.getenv("DEFAULT_GROUND_TRUTH_MIN_TITLE_LENGTH", "10")
)
DEFAULT_GROUND_TRUTH_QUESTION_COLUMN = "question"
DEFAULT_GROUND_TRUTH_ID_COLUMN = "source"

# Evaluation defaults
DEFAULT_TOP_K = 5
DEFAULT_TOKENIZER_MODEL = TokenizerModel.GPT_4O_MINI
DEFAULT_TOKENIZER_ENCODING_FALLBACK = TokenizerEncoding.CL100K_BASE
DEFAULT_SCORE_ALPHA = 2.0
DEFAULT_SCORE_BETA = 0.5
DEFAULT_SCORE_GAMMA = 1.5
DEFAULT_TOKEN_NORMALIZATION_DIVISOR = 1000.0
DEFAULT_SEARCH_TEXT_FIELDS = ["content", "title", "source"]
DEFAULT_CHUNK_TITLE = "Untitled"
DEFAULT_CHUNK_SOURCE = "Unknown"

# Search strategy configuration
DEFAULT_SEARCH_MODE = (
    SearchMode.EVALUATION
)  # Default to evaluation mode for consistent testing
MIN_SEARCH_CALLS = 3  # Minimum searches for consistency
MAX_SEARCH_CALLS = 8  # Maximum searches for cost control (production mode)
EARLY_STOP_CONFIDENCE_THRESHOLD = 0.9  # Confidence threshold for early stopping

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_RAG_MODEL = os.getenv("OPENAI_RAG_MODEL", str(DEFAULT_RAG_MODEL))
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", str(DEFAULT_JUDGE_MODEL))


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
