import pytest

from wikiagent.config import (
    MIN_GET_PAGE_CALLS,
    MIN_SEARCH_CALLS,
    TEST_QUESTIONS,
)
from wikiagent.wikipagent import query_wikipedia


@pytest.fixture
@pytest.mark.timeout(120)
@pytest.mark.parametrize("question", TEST_QUESTIONS)
async def agent_result(question):
    """Run Wikipedia agent query and return result for parametrized questions"""
    return await query_wikipedia(question)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("question", TEST_QUESTIONS)
async def test_agent_handles_different_questions(question):
    result = await query_wikipedia(question)

    # Verify basic structure
    assert result.answer is not None
    assert len(result.answer.answer) > 0
    assert len(result.tool_calls) > 0


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_search_tool_is_invoked(agent_result):
    tool_names = [call["tool_name"] for call in agent_result.tool_calls]
    assert "wikipedia_search" in tool_names, "Agent should invoke wikipedia_search tool"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_get_page_tool_is_invoked(agent_result):
    tool_names = [call["tool_name"] for call in agent_result.tool_calls]
    assert (
        "wikipedia_get_page" in tool_names
    ), "Agent should invoke wikipedia_get_page tool"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_search_tool_is_invoked_multiple_times(agent_result):
    """Test that wikipedia_search is invoked multiple times (Phase 1: 3-5, Phase 2: 8-12)"""
    search_calls = [
        call
        for call in agent_result.tool_calls
        if call["tool_name"] == "wikipedia_search"
    ]
    assert (
        len(search_calls) >= MIN_SEARCH_CALLS
    ), f"Agent should invoke wikipedia_search at least {MIN_SEARCH_CALLS} times, got {len(search_calls)}"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_get_page_tool_is_invoked_multiple_times(agent_result):
    get_page_calls = [
        call
        for call in agent_result.tool_calls
        if call["tool_name"] == "wikipedia_get_page"
    ]
    assert (
        len(get_page_calls) >= MIN_GET_PAGE_CALLS
    ), f"Agent should invoke wikipedia_get_page at least {MIN_GET_PAGE_CALLS} times, got {len(get_page_calls)}"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_agent_includes_sources(agent_result):
    assert len(agent_result.answer.sources_used) > 0, "Agent should include sources"
    assert all(isinstance(source, str) for source in agent_result.answer.sources_used)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_tool_calls_are_tracked(agent_result):
    assert len(agent_result.tool_calls) > 0, "Tool calls should be tracked"

    # Verify tool call structure
    for call in agent_result.tool_calls:
        assert "tool_name" in call
        assert "args" in call
        assert call["tool_name"] in {"wikipedia_search", "wikipedia_get_page"}
