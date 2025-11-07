# Tool functions for RAG Agent
"""Search tool function that agent can call repeatedly"""

from typing import List
from urllib.parse import quote

import requests

from rag_agent.models import (
    SearchResult,
    WikipediaPageContent,
    WikipediaSearchResult,
)

# Global search index (loaded once, used by tool)
_search_index = None
# Global tool call counter (incremented by agent's event handler)
_tool_call_count = 0
_max_tool_calls = 3  # Safety limit - can be overridden


def initialize_search_index(search_index) -> None:
    """
    Pre-load documents into search index.

    This should be called once before the agent starts making tool calls.

    Args:
        search_index: Initialized SearchIndex instance with documents loaded
    """
    global _search_index
    _search_index = search_index


def set_max_tool_calls(max_calls: int) -> None:
    """Set the maximum number of tool calls allowed"""
    global _max_tool_calls
    _max_tool_calls = max_calls


def reset_tool_call_count() -> None:
    """Reset the tool call counter (called at start of each query)"""
    global _tool_call_count
    _tool_call_count = 0


def search_documents(query: str, num_results: int = 2) -> List[SearchResult]:
    """
    Search the document index for relevant content.

    Use this tool to find information about user behavior patterns,
    questions, answers, and discussions from StackExchange.

    Args:
        query: Search query string (e.g., "user frustration", "satisfaction patterns")
        num_results: Number of results to return (default: 5)

    Returns:
        List of search results with content, source, similarity scores

    Raises:
        RuntimeError: If search index is not initialized or max tool calls exceeded
    """
    global _tool_call_count, _max_tool_calls

    if _search_index is None:
        raise RuntimeError(
            "Search index not initialized. Call initialize_search_index first."
        )

    _tool_call_count += 1
    if _tool_call_count > _max_tool_calls:
        raise RuntimeError(
            f"Maximum tool calls ({_max_tool_calls}) exceeded. "
            f"Please stop searching and provide your answer based on the previous searches."
        )

    results = _search_index.search(query=query, num_results=num_results)

    # Convert to SearchResult models
    search_results = []
    for doc in results:
        # Handle tags: convert string to list if needed (MinSearch returns tags as strings)
        tags = doc.get("tags", [])
        if isinstance(tags, str):
            # Convert space-separated string to list
            tags = [tag.strip() for tag in tags.split() if tag.strip()]
        elif not isinstance(tags, list):
            tags = []

        search_results.append(
            SearchResult(
                content=doc.get("content", ""),
                source=doc.get("source", "unknown"),
                title=doc.get("title"),
                similarity_score=doc.get("similarity_score"),
                tags=tags,
            )
        )

    return search_results


def wikipedia_search(query: str) -> List[WikipediaSearchResult]:
    """
    Search Wikipedia for pages matching the query.

    Use this tool to find relevant Wikipedia pages for a topic.
    The agent should use this first to discover which pages exist,
    then use wikipedia_get_page to retrieve the full content.

    Args:
        query: Search query string (e.g., "capybara", "Python programming")
               Spaces will be automatically converted to "+" for the API

    Returns:
        List of WikipediaSearchResult with title, snippet, page_id, etc.

    Raises:
        RuntimeError: If the API request fails or returns invalid data
    """
    try:
        # Replace spaces with "+" for Wikipedia API
        search_query = query.replace(" ", "+")

        url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={search_query}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes

        data = response.json()

        search_results = []
        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                search_results.append(
                    WikipediaSearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("snippet"),
                        page_id=item.get("pageid"),
                        size=item.get("size"),
                        word_count=item.get("wordcount"),
                    )
                )

        return search_results

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to search Wikipedia: {str(e)}")
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse Wikipedia API response: {str(e)}")


def wikipedia_get_page(title: str) -> WikipediaPageContent:
    """
    Get the raw wikitext content of a Wikipedia page.

    Use this tool to retrieve the full content of a Wikipedia page
    after using wikipedia_search to find the page title.

    Args:
        title: Wikipedia page title (e.g., "Capybara", "Python (programming language)")
               Spaces will be automatically converted to underscores for the URL

    Returns:
        WikipediaPageContent with title, content (raw wikitext), and URL

    Raises:
        RuntimeError: If the API request fails or page not found
    """
    try:
        # Replace spaces with underscores for Wikipedia URL
        page_title = title.replace(" ", "_")

        # URL encode the title for safety
        encoded_title = quote(page_title, safe="")

        url = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&action=raw"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        content = response.text

        wikipedia_url = f"https://en.wikipedia.org/wiki/{page_title}"

        return WikipediaPageContent(
            title=title,
            content=content,
            url=wikipedia_url,
        )

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to get Wikipedia page: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve Wikipedia page content: {str(e)}")
