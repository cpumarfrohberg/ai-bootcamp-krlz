import asyncio
import json
import logging
from typing import Any

import streamlit as st
from jaxn import StreamingJSONParser

from config import DEFAULT_SEARCH_MODE, LOG_LEVEL, SearchMode
from wikiagent.stream_handler import SearchAgentAnswerHandler
from wikiagent.wikipagent import query_wikipedia_stream

# Configure logging to display in terminal
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

QUERY_TRUNCATION_LENGTH = 50
MAX_QUESTION_LENGTH = 500
MODE_OPTIONS = ["evaluation", "production", "research"]

st.set_page_config(
    page_title="Wikipedia Agent",
    page_icon="🤖",
    layout="wide",
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("streaming", False)
st.session_state.setdefault("tool_calls", [])


def _get_mode_index(mode: SearchMode | None) -> int:
    """Get index for mode selectbox"""
    if mode and isinstance(mode, SearchMode):
        return MODE_OPTIONS.index(mode.value) if mode.value in MODE_OPTIONS else 0
    return 0


async def run_agent_stream(
    question: str,
    search_mode: SearchMode,
    answer_container: Any,
    confidence_container: Any,
    reasoning_container: Any,
    sources_container: Any,
    tool_calls_container: Any,
) -> None:
    handler = SearchAgentAnswerHandler(
        answer_container=answer_container,
        confidence_container=confidence_container,
        reasoning_container=reasoning_container,
        sources_container=sources_container,
    )
    handler.reset()
    parser = StreamingJSONParser(handler)
    tool_calls_list = []

    def _handle_tool_call(tool_name: str, args: str) -> None:
        try:
            args_dict = json.loads(args) if isinstance(args, str) else args
            query = (
                args_dict.get("query", "N/A")
                if isinstance(args_dict, dict)
                else str(args)
            )[:QUERY_TRUNCATION_LENGTH]
        except (json.JSONDecodeError, AttributeError, TypeError):
            query = str(args)[:QUERY_TRUNCATION_LENGTH] if args else "N/A"
        tool_calls_list.append({"tool_name": tool_name, "query": query})
        tool_calls_text = "\n".join(
            f"🔍 {i+1}. **{c['tool_name']}**: {c['query']}..."
            for i, c in enumerate(tool_calls_list)
        )
        tool_calls_container.markdown(tool_calls_text)

    def _handle_structured_output(delta: str) -> None:
        try:
            parser.parse_incremental(delta)
        except Exception:
            pass

    result = await query_wikipedia_stream(
        question=question,
        search_mode=search_mode,
        tool_call_callback=_handle_tool_call,
        structured_output_callback=_handle_structured_output,
    )
    st.session_state.last_result = result
    st.session_state.tool_calls = tool_calls_list


def main() -> None:
    st.title("🤖 Wikipedia Agent")

    with st.sidebar:
        st.header("Navigation")
        nav = st.radio(
            "Select a page", ["About", "Settings", "Chat"], label_visibility="visible"
        )
        st.divider()

    if nav == "About":
        _render_about_page()
    elif nav == "Settings":
        _render_settings_page()
    else:  # Chat
        _render_chat_page()


def _render_about_page() -> None:
    st.markdown("## Welcome to Wikipedia Agent")
    st.markdown(
        "Ask questions and get answers from Wikipedia with real-time streaming!"
    )
    st.divider()
    st.markdown("### How it works:")
    st.markdown(
        "1. Ask a question about any Wikipedia topic\n2. The agent searches Wikipedia\n3. Get real-time streaming answers with sources"
    )
    st.divider()
    st.markdown("### Search Modes:")
    st.markdown(
        "- **Evaluation**: Strict minimums for testing\n- **Production**: Adaptive search (recommended)\n- **Research**: Comprehensive search"
    )
    st.divider()
    st.markdown("### Example Questions:")
    for q in [
        "What is the capital of France?",
        "Who invented the telephone?",
        "Explain quantum computing",
    ]:
        st.markdown(f"- {q}")


def _render_settings_page() -> None:
    st.markdown("## Settings")
    current_mode = st.session_state.get("search_mode", DEFAULT_SEARCH_MODE)
    with st.form("settings_form"):
        mode_option = st.selectbox(
            "Search Mode", MODE_OPTIONS, index=_get_mode_index(current_mode)
        )
        show_reasoning = st.checkbox(
            "Show reasoning by default",
            value=st.session_state.get("show_reasoning_default", False),
        )
        show_sources = st.checkbox(
            "Show sources by default",
            value=st.session_state.get("show_sources_default", True),
        )
        if st.form_submit_button("Save Settings"):
            st.session_state.search_mode = SearchMode(mode_option.lower())
            st.session_state.show_reasoning_default = show_reasoning
            st.session_state.show_sources_default = show_sources
            st.success("Settings saved!")


def _render_chat_page() -> None:
    saved_mode = st.session_state.get("search_mode")
    with st.sidebar:
        st.header("Configuration")
        mode_option = st.selectbox(
            "Search Mode", MODE_OPTIONS, index=_get_mode_index(saved_mode)
        )
        search_mode = SearchMode(mode_option.lower())
        st.session_state.search_mode = search_mode
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.tool_calls = []
            st.session_state.last_result = None
            st.rerun()
        st.divider()
        st.header("Statistics")
        result = st.session_state.get("last_result")
        if result and hasattr(result, "usage") and result.usage:
            u = result.usage
            st.metric("Input Tokens", u.input_tokens)
            st.metric("Output Tokens", u.output_tokens)
            st.metric("Total Tokens", u.total_tokens)
        elif result and result.error:
            st.info("⚠️ Error occurred - check main chat for details")
        else:
            st.info("No queries yet. Ask a question to see token usage.")

    if not st.session_state.messages:
        st.info("👋 Start a conversation by asking a question about Wikipedia!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about Wikipedia..."):
        if len(prompt) > MAX_QUESTION_LENGTH:
            st.warning(
                "⚠️ Your question is quite long. Consider breaking it into smaller questions."
            )
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            tool_calls_container = st.empty()
            tool_calls_container.info("🤖 Agent is processing your question...")
            show_reasoning = st.session_state.get("show_reasoning_default", False)
            show_sources = st.session_state.get("show_sources_default", True)

            col1, col2 = st.columns([2, 1])
            with col1:
                answer_container = st.empty()
                sources_container = st.empty() if show_sources else None
            with col2:
                confidence_container = st.empty()
                reasoning_container = st.empty() if show_reasoning else None

            show_details = st.checkbox(
                "Show detailed information", value=show_reasoning or show_sources
            )
            with st.expander("📊 Detailed Information", expanded=show_details):
                detailed_reasoning_container = st.empty()
                detailed_sources_container = st.empty()
                tool_calls_detail_container = st.empty()

            st.session_state.streaming = True
            try:
                asyncio.run(
                    run_agent_stream(
                        prompt,
                        search_mode,
                        answer_container,
                        confidence_container,
                        reasoning_container,
                        sources_container,
                        tool_calls_container,
                    )
                )
                if "last_result" in st.session_state:
                    result = st.session_state.last_result

                    if result.error:
                        st.error(f"⚠️ {result.error.message}")
                        st.info(f"💡 {result.error.suggestion}")
                        if result.error.technical_details:
                            with st.expander("🔍 Technical Details"):
                                st.code(result.error.technical_details)
                    else:
                        if result.answer:
                            st.session_state.messages.append(
                                {"role": "assistant", "content": result.answer.answer}
                            )
                            st.session_state.last_search_mode = search_mode

                            if st.session_state.tool_calls:
                                tool_calls_text = "\n".join(
                                    f"🔍 {i+1}. **{c['tool_name']}**: {c['query']}..."
                                    for i, c in enumerate(st.session_state.tool_calls)
                                )
                                tool_calls_detail_container.markdown(
                                    f"**Tool Calls:**\n{tool_calls_text}"
                                )

                            if result.answer.reasoning:
                                detailed_reasoning_container.markdown(
                                    f"**Reasoning:**\n{result.answer.reasoning}"
                                )
                            if result.answer.sources_used:
                                detailed_sources_container.markdown(
                                    "**Sources:**\n"
                                    + "\n".join(
                                        f"- {s}" for s in result.answer.sources_used
                                    )
                                )
            finally:
                st.session_state.streaming = False

    if st.session_state.streaming:
        st.status("🔄 Streaming response...", state="running")
        st.caption(f"Search mode: {search_mode.value}")
    elif "last_result" not in st.session_state:
        st.status("✅ Ready to ask a question", state="complete")
    else:
        st.caption(
            f"Last query used {st.session_state.get('last_search_mode', search_mode).value} mode"
        )


if __name__ == "__main__":
    main()
