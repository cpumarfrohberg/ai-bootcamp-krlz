"""Tests for RAG Agent tools"""

import pytest
import requests

from config import SearchType
from rag_agent.models import SearchResult, WikipediaPageContent, WikipediaSearchResult
from rag_agent.tools import (
    initialize_search_index,
    search_documents,
    wikipedia_get_page,
    wikipedia_search,
)
from search.search_utils import SearchIndex

TEST_PAGE_TITLE = "Capybara"
TEST_PAGE_SNIPPET = "The capybara is the largest rodent..."
TEST_PAGE_CONTENT = "{{Infobox mammal\n| name = Capybara\n}}\nThe '''capybara'''..."
TEST_PAGE_ID = 12345
TEST_PAGE_SIZE = 50000
TEST_PAGE_WORD_COUNT = 2000
TEST_SEARCH_QUERY = "capybara"


@pytest.fixture
def mock_search_index():
    """Create a mock search index with sample documents"""
    index = SearchIndex(search_type=SearchType.MINSEARCH)

    # Note: tags must be a string for MinSearch (not a list)
    sample_docs = [
        {
            "content": "Users often express frustration when interfaces are slow to respond.",
            "source": "question_123",
            "title": "Why do users get frustrated?",
            "tags": "user-behavior frustration",  # Converted to string
        },
        {
            "content": "Satisfaction is measured through user feedback and surveys.",
            "source": "question_456",
            "title": "Measuring user satisfaction",
            "tags": "user-behavior satisfaction",  # Converted to string
        },
        {
            "content": "Usability issues include confusing navigation and unclear labels.",
            "source": "question_789",
            "title": "Common usability problems",
            "tags": "usability user-experience",  # Converted to string
        },
    ]

    index.add_documents(sample_docs)
    return index


def test_initialize_search_index(mock_search_index):
    """Test initialize_search_index sets global index"""
    initialize_search_index(mock_search_index)

    # Verify we can call search_documents now
    results = search_documents("user frustration", num_results=2)

    assert len(results) > 0
    assert isinstance(results[0], SearchResult)


def test_search_documents_returns_results(mock_search_index):
    """Test search_documents returns SearchResult objects"""
    initialize_search_index(mock_search_index)

    results = search_documents("user frustration", num_results=2)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(result, SearchResult) for result in results)

    # Verify SearchResult structure
    result = results[0]
    assert result.content is not None
    assert result.source is not None


def test_search_documents_handles_empty_index():
    """Test search_documents handles uninitialized index"""
    # Import the global _search_index to reset it
    # Temporarily set to None to test error handling
    import rag_agent.tools as tools_module
    from rag_agent.tools import _search_index

    original_index = tools_module._search_index
    tools_module._search_index = None

    try:
        with pytest.raises(RuntimeError, match="Search index not initialized"):
            search_documents("test query", num_results=1)
    finally:
        # Restore original index
        tools_module._search_index = original_index


def test_search_documents_num_results(mock_search_index):
    """Test search_documents respects num_results parameter"""
    initialize_search_index(mock_search_index)

    results_3 = search_documents("user", num_results=3)
    results_1 = search_documents("user", num_results=1)

    assert len(results_3) <= 3
    assert len(results_1) <= 1


def test_search_documents_result_structure(mock_search_index):
    """Test search_documents returns properly structured results"""
    initialize_search_index(mock_search_index)

    results = search_documents("satisfaction", num_results=2)

    for result in results:
        assert isinstance(result, SearchResult)
        assert result.content is not None
        assert isinstance(result.content, str)
        assert len(result.content) > 0


def test_wikipedia_search_success(mocker):
    """Test wikipedia_search returns results on successful API call"""
    mock_get = mocker.patch("rag_agent.tools.requests.get")
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
    mock_get = mocker.patch("rag_agent.tools.requests.get")
    mock_get.side_effect = requests.RequestException("Network error")

    with pytest.raises(RuntimeError, match="Failed to search Wikipedia"):
        wikipedia_search(TEST_SEARCH_QUERY)


def test_wikipedia_get_page_success(mocker):
    """Test wikipedia_get_page returns page content on successful API call"""
    mock_get = mocker.patch("rag_agent.tools.requests.get")
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
    mock_get = mocker.patch("rag_agent.tools.requests.get")
    mock_get.side_effect = requests.RequestException("Network error")

    with pytest.raises(RuntimeError, match="Failed to get Wikipedia page"):
        wikipedia_get_page(TEST_PAGE_TITLE)
