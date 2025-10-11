# Text-based RAG implementation for GitHub repository analysis
from typing import Callable, List

from minsearch import Index

from chunking.utils import chunk_documents
from config import DEFAULT_INSTRUCTION_TYPE, openai_client
from github.parser import parse_data
from github.reader import read_github_data
from rag.utils import answer_question_with_context


class TextRAG:
    """Text-based RAG implementation for GitHub repository analysis using minsearch"""

    def __init__(self, text_fields: List[str] | None = None):
        self.text_fields = text_fields or [
            "content",
            "filename",
            "title",
            "description",
        ]
        self.documents = []
        self.chunks = []
        self.index = None

    def load_repository(
        self,
        repo_owner: str,
        repo_name: str,
        allowed_extensions: set | None = None,
        filename_filter: Callable | None = None,
    ):
        """Load and process GitHub repository data"""

        # Step 1: Fetch GitHub data
        github_data = read_github_data(
            repo_owner=repo_owner,
            repo_name=repo_name,
            allowed_extensions=allowed_extensions,
            filename_filter=filename_filter,
        )

        # Step 2: Parse the data
        parsed_data = parse_data(github_data)

        # Step 3: Chunk the documents
        self.chunks = chunk_documents(parsed_data)

        # Step 4: Create minsearch index
        self.index = Index(text_fields=self.text_fields)
        self.index.fit(self.chunks)

        self.documents = parsed_data

    def ask(
        self, question: str, instruction_type: str = DEFAULT_INSTRUCTION_TYPE
    ) -> str:
        """Ask a question using text-based search"""
        if not self.index:
            raise ValueError("No repository loaded. Call load_repository() first.")

        return answer_question_with_context(
            question=question,
            index=self.index,
            instruction_type=instruction_type,
            openai_client=openai_client,
        )
