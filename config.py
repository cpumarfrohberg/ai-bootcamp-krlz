import os
from enum import Enum
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class InstructionType(Enum):
    FAQ_ASSISTANT = "faq_assistant"
    TECHNICAL_SUPPORT = "technical_support"


class PromptTemplate(Enum):
    FAQ_ASSISTANT = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

    TECHNICAL_SUPPORT = """
<QUESTION>
{question}
</QUESTION>

<TECHNICAL_CONTEXT>
{context}
</TECHNICAL_CONTEXT>

Please provide a clear, step-by-step solution based on the technical context above.
""".strip()


class InstructionsConfig:
    INSTRUCTIONS: Dict[InstructionType, str] = {
        InstructionType.FAQ_ASSISTANT: """
You're a helpful FAQ assistant. Answer questions based on the provided CONTEXT.
Be concise, accurate, and helpful. If you don't know the answer based on the context, say so.
""".strip(),
        InstructionType.TECHNICAL_SUPPORT: """
You're a technical support assistant. Help users with technical questions.
Provide clear, step-by-step solutions when possible.
Use the CONTEXT to provide accurate technical information.
""".strip(),
    }


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_INSTRUCTION_TYPE = InstructionType.FAQ_ASSISTANT

# GitHub repository configuration
DEFAULT_REPO_OWNER = "evidentlyai"
DEFAULT_REPO_NAME = "docs"
DEFAULT_ALLOWED_EXTENSIONS = {"md", "mdx"}

# Document chunking configuration
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_STEP = 1000
DEFAULT_CONTENT_FIELD = "content"

# Search configuration
DEFAULT_NUM_RESULTS = 5
DEFAULT_BOOST_DICT = {"question": 3.0, "section": 0.3}
DEFAULT_FILTER_DICT = {"course": "data-engineering-zoomcamp"}

# OpenAI configuration
API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=API_KEY)
