# Pydantic models for RAG Agent

from pydantic import BaseModel, Field

# Constants for validation
MIN_CONFIDENCE_SCORE = 0.0
MAX_CONFIDENCE_SCORE = 1.0
MIN_TOKEN_COUNT = 0
MIN_JUDGE_SCORE = 0.0
MAX_JUDGE_SCORE = 1.0


class SearchResult(BaseModel):
    content: str = Field(..., description="The content/text of the search result")
    filename: str | None = Field(None, description="Filename or source identifier")
    title: str | None = Field(None, description="Title of the document")
    similarity_score: float | None = Field(
        None, description="Relevance score from search"
    )
    source: str | None = Field(None, description="Source of the document")
    tags: list[str] | None = Field(
        None, description="Tags associated with the document"
    )


class SearchAgentAnswer(BaseModel):
    answer: str = Field(..., description="The answer to the user's question")
    confidence: float = Field(
        ...,
        ge=MIN_CONFIDENCE_SCORE,
        le=MAX_CONFIDENCE_SCORE,
        description=f"Confidence in the answer ({MIN_CONFIDENCE_SCORE} to {MAX_CONFIDENCE_SCORE})",
    )
    sources_used: list[str] = Field(
        ..., description="List of source filenames used to generate the answer"
    )
    reasoning: str | None = Field(
        None, description="Brief explanation of the reasoning behind the answer"
    )


class WikipediaSearchResult(BaseModel):
    title: str = Field(..., description="Wikipedia page title")
    snippet: str | None = Field(None, description="Text snippet from the page")
    page_id: int | None = Field(None, description="Wikipedia page ID")
    size: int | None = Field(None, description="Page size in bytes")
    word_count: int | None = Field(None, description="Word count of the page")


class WikipediaPageContent(BaseModel):
    title: str = Field(..., description="Wikipedia page title")
    content: str = Field(..., description="Raw wikitext content")
    url: str | None = Field(None, description="Full Wikipedia URL")


class TokenUsage(BaseModel):
    """Token usage information from LLM API calls"""

    input_tokens: int = Field(
        ..., ge=MIN_TOKEN_COUNT, description="Number of input tokens used"
    )
    output_tokens: int = Field(
        ..., ge=MIN_TOKEN_COUNT, description="Number of output tokens used"
    )
    total_tokens: int = Field(..., ge=MIN_TOKEN_COUNT, description="Total tokens used")


class AgentError(BaseModel):
    """Structured error information from agent execution"""

    error_type: str = Field(
        ..., description="Type of error (e.g., 'WikipediaAPI', 'Network', 'Timeout')"
    )
    message: str = Field(..., description="User-friendly error message")
    suggestion: str = Field(..., description="Suggested action for the user")
    technical_details: str | None = Field(
        None, description="Technical error details for debugging"
    )


class WikipediaAgentResponse(BaseModel):
    answer: SearchAgentAnswer | None = Field(
        None, description="The structured answer (None if error occurred)"
    )
    tool_calls: list[dict] = Field(
        default_factory=list, description="List of tool calls made during query"
    )
    usage: TokenUsage | None = Field(
        None, description="Token usage information (None if error occurred)"
    )
    error: AgentError | None = Field(
        None, description="Error information if query failed"
    )


class JudgeEvaluation(BaseModel):
    """Judge evaluation output for answer quality assessment"""

    overall_score: float = Field(
        ...,
        ge=MIN_JUDGE_SCORE,
        le=MAX_JUDGE_SCORE,
        description=f"Overall quality score ({MIN_JUDGE_SCORE} to {MAX_JUDGE_SCORE})",
    )
    accuracy: float = Field(
        ...,
        ge=MIN_JUDGE_SCORE,
        le=MAX_JUDGE_SCORE,
        description=f"Factual correctness score ({MIN_JUDGE_SCORE} to {MAX_JUDGE_SCORE})",
    )
    completeness: float = Field(
        ...,
        ge=MIN_JUDGE_SCORE,
        le=MAX_JUDGE_SCORE,
        description=f"Answer thoroughness score ({MIN_JUDGE_SCORE} to {MAX_JUDGE_SCORE})",
    )
    relevance: float = Field(
        ...,
        ge=MIN_JUDGE_SCORE,
        le=MAX_JUDGE_SCORE,
        description=f"Answer relevance to question ({MIN_JUDGE_SCORE} to {MAX_JUDGE_SCORE})",
    )
    reasoning: str = Field(..., description="Brief explanation of the evaluation")


class JudgeResult(BaseModel):
    """Result from judge evaluation including evaluation and usage"""

    evaluation: JudgeEvaluation = Field(..., description="Judge evaluation scores")
    usage: TokenUsage = Field(..., description="Token usage information")
