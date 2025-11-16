# Tool functions for Wikipedia Agent
"""Wikipedia API tools for searching and retrieving page content"""

import logging
from typing import List
from urllib.parse import quote

import requests

from wikiagent.config import MAX_PAGE_CONTENT_LENGTH, USER_AGENT
from wikiagent.models import WikipediaPageContent, WikipediaSearchResult

logger = logging.getLogger(__name__)


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

        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
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
        RuntimeError: If network error or HTTP error other than 404 occurs
    """
    # Replace spaces with underscores for Wikipedia URL (used in all code paths)
    page_title = title.replace(" ", "_")

    try:
        # URL encode the title for safety
        encoded_title = quote(page_title, safe="")

        url = f"https://en.wikipedia.org/w/index.php?title={encoded_title}&action=raw"

        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        content = response.text

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

    except requests.HTTPError as e:
        # Handle 404 gracefully (page doesn't exist - agent can continue)
        if e.response and e.response.status_code == 404:
            logger.warning(f"Wikipedia page not found: {title} (404)")
            # Return empty content with a note - agent can continue
            return WikipediaPageContent(
                title=title,
                content=f"[Page not found: {title} does not exist on Wikipedia]",
                url=f"https://en.wikipedia.org/wiki/{page_title}",
            )
        else:
            # Other HTTP errors (500, 503, etc.) - raise exception (server issues)
            status_code = e.response.status_code if e.response else "unknown"
            logger.error(f"HTTP error retrieving Wikipedia page {title}: {status_code}")
            raise RuntimeError(
                f"Failed to get Wikipedia page {title}: HTTP {status_code}"
            )
    except requests.RequestException as e:
        # Network errors, timeouts, connection issues - raise exception (infrastructure problems)
        logger.error(f"Network error retrieving Wikipedia page {title}: {str(e)}")
        raise RuntimeError(f"Failed to get Wikipedia page {title}: {str(e)}")
    except Exception as e:
        # Any other unexpected errors - log and return empty
        logger.warning(f"Unexpected error retrieving Wikipedia page {title}: {str(e)}")
        return WikipediaPageContent(
            title=title,
            content=f"[Error retrieving page: {title} - {str(e)}]",
            url=f"https://en.wikipedia.org/wiki/{page_title}",
        )
