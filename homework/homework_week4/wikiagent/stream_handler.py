"""Streaming JSON parser handler for SearchAgentAnswer structured output"""

from jaxn import JSONParserHandler


class SearchAgentAnswerHandler(JSONParserHandler):
    def __init__(
        self,
        answer_container: object | None = None,
        confidence_container: object | None = None,
        reasoning_container: object | None = None,
        sources_container: object | None = None,
    ):
        """
        Initialize handler with Streamlit containers for UI updates.

        Args:
            answer_container: st.empty() container for streaming answer text
            confidence_container: st.empty() container for confidence metric
            reasoning_container: st.empty() container for reasoning text
            sources_container: st.empty() container for sources list
        """
        super().__init__()
        self.answer_container = answer_container
        self.confidence_container = confidence_container
        self.reasoning_container = reasoning_container
        self.sources_container = sources_container

        # Track state for incremental updates
        self.current_answer = ""
        self.current_confidence: float | None = None
        self.current_reasoning: str | None = None
        self.sources_list: list[str] = []

    def reset(self) -> None:
        """Reset handler state for a new query"""
        self.current_answer = ""
        self.current_confidence = None
        self.current_reasoning = None
        self.sources_list = []

    def on_field_start(self, path: str, field_name: str) -> None:
        """Called when starting to read a field value"""
        # Initialize arrays when starting
        if field_name == "sources_used" and path == "":
            self.sources_list = []

    def on_field_end(
        self,
        path: str,
        field_name: str,
        value: object,
        parsed_value: object | None = None,
    ) -> None:
        """
        Called when a field value is complete.
        Update Streamlit UI components when fields finish.
        """
        if field_name == "answer" and path == "":
            # Ensure answer is fully displayed when field completes
            if self.answer_container and self.current_answer:
                self.answer_container.markdown(self.current_answer)

        elif field_name == "confidence" and path == "":
            # Display confidence as a metric
            self.current_confidence = float(value) if value is not None else None
            if self.confidence_container and self.current_confidence is not None:
                self.confidence_container.metric(
                    "Confidence", f"{self.current_confidence:.2%}"
                )

        elif field_name == "reasoning" and path == "":
            # Display reasoning when complete
            self.current_reasoning = str(value) if value is not None else None
            if self.reasoning_container and self.current_reasoning:
                self.reasoning_container.markdown(
                    f"**Reasoning:** {self.current_reasoning}"
                )

    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        """
        Called for each character as string values stream in.
        Stream answer content as it arrives.
        """
        if field_name == "answer" and path == "":
            # Accumulate answer text
            self.current_answer += chunk
            # Update Streamlit container with current answer
            if self.answer_container:
                self.answer_container.markdown(self.current_answer)

    def on_array_item_end(
        self, path: str, field_name: str, item: object | None = None
    ) -> None:
        """
        Called when finishing an object in an array.
        Display sources as they complete.
        """
        if field_name != "sources_used" or path != "" or item is None:
            return

        # Sources are strings in the array
        source = str(item).strip('"')
        if not source or source in self.sources_list:
            return

        self.sources_list.append(source)
        if self.sources_container:
            sources_text = "\n".join(f"- {s}" for s in self.sources_list)
            self.sources_container.markdown(f"**Sources:**\n{sources_text}")
