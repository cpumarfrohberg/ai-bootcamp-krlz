"""Tests for RAG Agent tools"""

import pytest
import requests

from wikiagent.models import WikipediaPageContent, WikipediaSearchResult
from wikiagent.tools import wikipedia_get_page, wikipedia_search

TEST_PAGE_TITLE = "Consumer behaviour"
TEST_PAGE_SNIPPET = (
    "Consumer behaviour is the study of factors that influence customer decisions..."
)
TEST_PAGE_CONTENT = "{{Infobox topic\n| name = Consumer behaviour\n}}\n'''Consumer behaviour''' is the study of factors..."
TEST_PAGE_ID = 12345
TEST_PAGE_SIZE = 50000
TEST_PAGE_WORD_COUNT = 2000
TEST_SEARCH_QUERY = "customer behavior"


def test_wikipedia_search_success(mocker):
    """Test wikipedia_search returns results on successful API call"""
    mock_get = mocker.patch("wikiagent.tools.requests.get")
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "query": {
            "search": [
                {
                    "title": TEST_PAGE_TITLE,
                    "snippet": TEST_PAGE_SNIPPET,
                    "pageid": TEST_PAGE_ID,
                    "size": TEST_PAGE_SIZE,
                    "wordcount": TEST_PAGE_WORD_COUNT,
                }
            ]
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    results = wikipedia_search(TEST_SEARCH_QUERY)

    assert len(results) == 1
    assert isinstance(results[0], WikipediaSearchResult)
    assert results[0].title == TEST_PAGE_TITLE
    assert results[0].snippet == TEST_PAGE_SNIPPET
    assert results[0].page_id == TEST_PAGE_ID
    assert results[0].size == TEST_PAGE_SIZE
    assert results[0].word_count == TEST_PAGE_WORD_COUNT


def test_wikipedia_search_handles_network_error(mocker):
    """Test wikipedia_search handles network errors"""
    mock_get = mocker.patch("wikiagent.tools.requests.get")
    mock_get.side_effect = requests.RequestException("Network error")

    with pytest.raises(RuntimeError, match="Failed to search Wikipedia"):
        wikipedia_search(TEST_SEARCH_QUERY)


def test_wikipedia_get_page_success(mocker):
    """Test wikipedia_get_page returns page content on successful API call"""
    mock_get = mocker.patch("wikiagent.tools.requests.get")
    mock_response = mocker.Mock()
    mock_response.text = TEST_PAGE_CONTENT
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = wikipedia_get_page(TEST_PAGE_TITLE)

    assert isinstance(result, WikipediaPageContent)
    assert result.title == TEST_PAGE_TITLE
    assert result.content == TEST_PAGE_CONTENT
    assert (
        result.url
        == f"https://en.wikipedia.org/wiki/{TEST_PAGE_TITLE.replace(' ', '_')}"
    )


def test_wikipedia_get_page_handles_network_error(mocker):
    """Test wikipedia_get_page handles network errors"""
    mock_get = mocker.patch("wikiagent.tools.requests.get")
    mock_get.side_effect = requests.RequestException("Network error")

    with pytest.raises(RuntimeError, match="Failed to get Wikipedia page"):
        wikipedia_get_page(TEST_PAGE_TITLE)
