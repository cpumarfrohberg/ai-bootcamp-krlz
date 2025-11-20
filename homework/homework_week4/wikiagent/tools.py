from typing import List
from urllib.parse import quote

import requests
from pydantic import ValidationError

from wikiagent.config import (
    MAX_PAGE_CONTENT_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_TITLE_LENGTH,
    USER_AGENT,
)
from wikiagent.models import WikipediaPageContent, WikipediaSearchResult


def _validate_search_query(query: str) -> None:
    """Validate search query input before API call"""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Search query too long (max {MAX_QUERY_LENGTH} chars)")


def _validate_page_title(title: str) -> None:
    """Validate page title input before API call"""
    if not title or not title.strip():
        raise ValueError("Page title cannot be empty")
    if len(title) > MAX_TITLE_LENGTH:
        raise ValueError(f"Page title too long (max {MAX_TITLE_LENGTH} chars)")


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
        ValueError: If input validation fails
        RuntimeError: If the API request fails or returns invalid data
    """
    _validate_search_query(query)

    try:
        search_query = query.replace(" ", "+")

        url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={search_query}"

        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        search_results = []
        if "query" in data and "search" in data["query"]:
            search_items = data["query"]["search"]
            if not isinstance(search_items, list):
                raise RuntimeError("Invalid API response: 'search' is not a list")

            for item in search_items:
                try:
                    result = WikipediaSearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("snippet"),
                        page_id=item.get("pageid"),
                        size=item.get("size"),
                        word_count=item.get("wordcount"),
                    )
                    search_results.append(result)
                except ValidationError:
                    continue

        return search_results

    except ValueError:
        raise
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to search Wikipedia: {str(e)}")
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse Wikipedia API response: {str(e)}")


def wikipedia_get_page(title: str) -> WikipediaPageContent:
    """
    Get the raw wikitext content of a Wikipedia page.

    Use this tool to retrieve the full content of a Wikipedia page
    after using wikipedia_search to find the page title.

    Error handling:
    - 404 (page not found): Returns WikipediaPageContent with error message
      (allows agent to continue processing other pages)
    - Network errors (connection, timeout): Raises RuntimeError
    - Other HTTP errors (500, 503): Raises RuntimeError

    Args:
        title: Wikipedia page title (e.g., "Capybara", "Python (programming language)")
               Spaces will be automatically converted to underscores for the URL

    Returns:
        WikipediaPageContent with title, content (raw wikitext), and URL.
        If page not found (404), content will contain an error message.

    Raises:
        ValueError: If input validation fails
        RuntimeError: If network error or HTTP error other than 404 occurs
    """
    _validate_page_title(title)
    page_title = title.replace(" ", "_")

    try:
        # URL encode the title for safety
        encoded_title = quote(page_title, safe="")

        url = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&action=raw"

        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        content = response.text
        if not content:
            raise RuntimeError(f"Empty content received for page: {title}")

        # Truncate content to prevent context overflow
        # Keep the beginning of the page which usually contains the most relevant information
        if len(content) > MAX_PAGE_CONTENT_LENGTH:
            content = content[:MAX_PAGE_CONTENT_LENGTH]
            # Add truncation indicator
            content += f"\n\n[Content truncated to {MAX_PAGE_CONTENT_LENGTH} characters. Full page available at: https://en.wikipedia.org/wiki/{page_title}]"

        wikipedia_url = f"https://en.wikipedia.org/wiki/{page_title}"

        return WikipediaPageContent(
            title=title,
            content=content,
            url=wikipedia_url,
        )

    except ValueError:
        raise
    except requests.HTTPError as e:
        if e.response and e.response.status_code == 404:
            return WikipediaPageContent(
                title=title,
                content=f"[Page not found: {title} does not exist on Wikipedia]",
                url=f"https://en.wikipedia.org/wiki/{page_title}",
            )
        else:
            status_code = e.response.status_code if e.response else "unknown"
            raise RuntimeError(
                f"Failed to get Wikipedia page {title}: HTTP {status_code}"
            )
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to get Wikipedia page {title}: {str(e)}")
    except ValidationError as e:
        raise RuntimeError(f"Invalid data received for page {title}: {str(e)}")
    except Exception as e:
        return WikipediaPageContent(
            title=title,
            content=f"[Error retrieving page: {title} - {str(e)}]",
            url=f"https://en.wikipedia.org/wiki/{page_title}",
        )
