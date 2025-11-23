import asyncio
import json
import logging
from typing import Any

import streamlit as st
from jaxn import StreamingJSONParser

from config import DEFAULT_SEARCH_MODE, LOG_LEVEL, SearchMode
from wikiagent.monitoring.db import get_cost_stats, get_recent_logs, init_db
from wikiagent.stream_handler import SearchAgentAnswerHandler
from wikiagent.wikipagent import query_wikipedia_stream

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

MODE_OPTIONS = ["evaluation", "production", "research"]

st.set_page_config(page_title="Wikipedia Agent", page_icon="🤖", layout="wide")

# Initialize session state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("db_initialized", False)

# Initialize database
if not st.session_state.db_initialized:
    init_db()
    st.session_state.db_initialized = True


async def run_agent_stream(
    question: str,
    search_mode: SearchMode,
    answer_container: Any,
    tool_calls_container: Any,
) -> None:
    """Run agent with streaming."""
    handler = SearchAgentAnswerHandler(
        answer_container=answer_container,
        confidence_container=None,
        reasoning_container=None,
        sources_container=None,
    )
    handler.reset()
    parser = StreamingJSONParser(handler)

    def _handle_tool_call(tool_name: str, args: str) -> None:
        try:
            args_dict = json.loads(args) if isinstance(args, str) else args
            query = (
                args_dict.get("query", "N/A")
                if isinstance(args_dict, dict)
                else str(args)
            )
        except (json.JSONDecodeError, AttributeError, TypeError):
            query = str(args) if args else "N/A"
        tool_calls_container.markdown(f"🔍 **{tool_name}**: {query[:50]}...")

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


def main() -> None:
    st.title("🤖 Wikipedia Agent")

    with st.sidebar:
        st.header("Navigation")
        nav = st.radio(
            "Page", ["Chat", "Monitoring", "About"], label_visibility="collapsed"
        )
        st.divider()

        if nav == "Chat":
            st.header("Configuration")
            mode_option = st.selectbox(
                "Search Mode",
                MODE_OPTIONS,
                index=MODE_OPTIONS.index(
                    st.session_state.get("search_mode", DEFAULT_SEARCH_MODE).value
                ),
            )
            st.session_state.search_mode = SearchMode(mode_option.lower())
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.last_result = None
                st.rerun()

    if nav == "About":
        _render_about_page()
    elif nav == "Monitoring":
        _render_monitoring_page()
    else:  # Chat
        _render_chat_page()


def _render_about_page() -> None:
    st.markdown("## Welcome to Wikipedia Agent")
    st.markdown(
        "Ask questions and get answers from Wikipedia with real-time streaming!"
    )
    st.markdown("### How it works:")
    st.markdown("1. Ask a question about any Wikipedia topic")
    st.markdown("2. The agent searches Wikipedia")
    st.markdown("3. Get real-time streaming answers with sources")


def _render_chat_page() -> None:
    search_mode = st.session_state.get("search_mode", DEFAULT_SEARCH_MODE)

    if not st.session_state.messages:
        st.info("👋 Ask a question about Wikipedia!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about Wikipedia..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer_container = st.empty()
            tool_calls_container = st.empty()
            tool_calls_container.info("🤖 Processing...")

            try:
                asyncio.run(
                    run_agent_stream(
                        prompt, search_mode, answer_container, tool_calls_container
                    )
                )
                result = st.session_state.get("last_result")

                if result and result.error:
                    st.error(f"⚠️ {result.error.message}")
                elif result and result.answer:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result.answer.answer}
                    )
                    if result.answer.sources_used:
                        st.markdown(
                            "**Sources:** " + ", ".join(result.answer.sources_used)
                        )
            except Exception as e:
                st.error(f"Error: {e}")


def _render_monitoring_page() -> None:
    st.header("📊 Monitoring Dashboard")

    # Cost Statistics
    st.subheader("💰 Cost Statistics")
    try:
        stats = get_cost_stats()
        if stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Cost", f"${stats['total_cost']:.4f}")
            col2.metric("Total Queries", stats["total_queries"])
            col3.metric("Average Cost", f"${stats['avg_cost']:.4f}")
        else:
            st.info("No cost data available yet.")
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

    st.divider()

    # Recent Logs
    st.subheader("📝 Recent Logs")
    try:
        logs = get_recent_logs(limit=20)
        if logs:
            for log in logs:
                with st.expander(
                    f"Log #{log.get('id')} - {log.get('user_prompt', 'N/A')[:50]}..."
                ):
                    st.write(f"**Time:** {log.get('created_at')}")
                    st.write(f"**Model:** {log.get('model', 'N/A')}")
                    st.write(
                        f"**Cost:** ${log.get('total_cost', 0):.4f}"
                        if log.get("total_cost")
                        else "**Cost:** N/A"
                    )
                    st.write(
                        f"**Tokens:** {log.get('total_input_tokens', 'N/A')} in / {log.get('total_output_tokens', 'N/A')} out"
                    )
        else:
            st.info("No logs available yet.")
    except Exception as e:
        st.error(f"Error loading logs: {e}")


if __name__ == "__main__":
    main()
