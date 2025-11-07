# Pydantic models for RAG Agent

from pydantic import BaseModel, Field


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


class RAGAnswer(BaseModel):
    answer: str = Field(..., description="The answer to the user's question")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the answer (0.0 to 1.0)"
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
