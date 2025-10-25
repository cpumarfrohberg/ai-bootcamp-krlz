import os
from enum import Enum
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class InstructionType(Enum):
    PODCAST_ASSISTANT = "podcast_assistant"


class PromptTemplate(Enum):
    PODCAST_ASSISTANT = """
<QUESTION>
{question}
</QUESTION>

<PODCAST_CONTEXT>
{context}
</PODCAST_CONTEXT>

Please provide a helpful answer based on the podcast content above. Include timestamps when relevant.
""".strip()


class InstructionsConfig:
    INSTRUCTIONS: Dict[InstructionType, str] = {
        InstructionType.PODCAST_ASSISTANT: """
You're a helpful podcast assistant. Answer questions based on the provided PODCAST_CONTEXT.
Be concise, accurate, and helpful. When relevant, include timestamps from the podcast content.
If you don't know the answer based on the context, say so.
""".strip(),
    }


class ModelType(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class ChunkingConfig(Enum):
    DEFAULT_SIZE = 2000
    DEFAULT_OVERLAP = 0.5
    DEFAULT_CONTENT_FIELD = "content"


class RepositoryConfig(Enum):
    DEFAULT_OWNER = "DataTalksClub"
    DEFAULT_NAME = "datatalksclub.github.io"
    DEFAULT_EXTENSIONS = {"md", "mdx"}
    GITHUB_CODELOAD_URL = "https://codeload.github.com"


class SearchType(Enum):
    TEXT = "text"
    VECTOR_MINSEARCH = "vector_minsearch"
    VECTOR_SENTENCE_TRANSFORMERS = "vector_sentence_transformers"


class SentenceTransformerModel(Enum):
    # Most commonly used models
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"
    PARAPHRASE_MULTILINGUAL_MINI = "paraphrase-multilingual-MiniLM-L12-v2"


class GitHubConfig(Enum):
    BASE_URL = "https://codeload.github.com"
    TIMEOUT = 30
    MAX_FILE_SIZE = 10_000_000
    MAX_FILES = 1000
    MAX_TOTAL_SIZE = 50_000_000


class FileProcessingConfig(Enum):
    MAX_FILE_SIZE = 10_000_000  # 10MB per file
    MAX_CONTENT_SIZE = 5_000_000  # 5MB content
    ALLOWED_EXTENSIONS = {"md", "mdx", "txt", "rst", "adoc"}
    BLOCKED_EXTENSIONS = {"exe", "bat", "sh", "py", "js", "jar", "dll", "so"}


class PodcastConstants(Enum):
    """Constants specific to podcast processing"""

    # Time conversion constants
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600

    # Text processing constants
    MAX_TEXT_PREVIEW_LENGTH = 50
    MAX_CONTEXT_LENGTH = 500
    DEFAULT_FALLBACK_CONFIDENCE = 0.5

    # Search constants
    DEFAULT_SEARCH_RESULTS_LIMIT = 3
    DEFAULT_VECTOR_SEARCH_RESULTS = 5

    # Confidence range constants
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0


# OpenAI configuration
API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=API_KEY)
