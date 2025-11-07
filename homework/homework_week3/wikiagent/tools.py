# Tool functions for Wikipedia Agent
"""Wikipedia API tools for searching and retrieving page content"""

from typing import List
from urllib.parse import quote

import requests

from wikiagent.models import WikipediaPageContent, WikipediaSearchResult


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
