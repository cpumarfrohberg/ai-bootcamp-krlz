# Podcast Assistant

A RAG (Retrieval-Augmented Generation) powered assistant for podcast content analysis and question answering.

## Features

- **Podcast Processing**: Specialized parsing of podcast transcripts with timestamps
- **GitHub Integration**: Parse and analyze GitHub repositories containing podcast content
- **RAG Pipeline**: Advanced retrieval and generation capabilities with multiple search types
- **CLI Interface**: Easy-to-use command-line interface
- **Vector Search**: SentenceTransformers-based semantic search
- **Structured Output**: Confidence scores, source tracking, and reasoning
- **Security**: Enterprise-level file validation and processing limits

## Prerequisites

- Python 3.11+
- Git
- uv (recommended) or pip

## Run cli in container

```bash
# 1. Run Docker in one terminal session
docker compose up --build

# 2. Wait until you read "podcast-assistant-redis  | 1:M 16 Oct 2025 21:03:27.285 * Ready to accept connections tcp", then run cli in a separate terminal (see below for more run commands)
uv run ra --help

# 3. Stop by pressing `control + c`

# 4. Remove clutter
docker compose down
```

## Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## CLI Usage

The CLI provides a simple interface to query GitHub repositories containing podcast content using RAG.

### Basic Usage

```bash
# Ask a question about the default repository (using default chunking, fast and lightweight)
uv run ra "What's the first episode in the results for 'how do I make money with AI?'"

# Ask with custom chunking parameters
uv run ra "What's the first episode in the results for 'how do I make money with AI?'" 1500 0.75
```

### Command Parameters

The `ra` command accepts the following parameters:

#### Required Parameters
- **`question`** (positional argument): The question you want to ask about the repository
- **`chunk_size`** (positional argument): Size of document chunks in characters (default: `30`)
- **`overlap`** (positional argument): Overlap ratio between chunks (default: `0.5`)

#### Optional Parameters

- **`--owner`, `-o`**: GitHub repository owner (default: `DataTalksClub`)
- **`--repo`, `-r`**: GitHub repository name (default: `datatalksclub.github.io`)
- **`--extensions`, `-e`**: Comma-separated file extensions to include (default: `md,mdx`)
- **`--search-type`, `-s`**: Search type: `text`, `vector_minsearch`, `vector_sentence_transformers` (default: `text`)
- **`--model`, `-m`**: SentenceTransformer model name (only for `vector_sentence_transformers`)
- **`--verbose`, `-v`**: Enable verbose output for debugging

### Examples

#### Text Search Examples
```bash
# Text search with verbose output
uv run ra "What's the first episode in the results for 'how do I make money with AI?'" --verbose
```

#### Vector Search Examples
```bash
# Vector search with default model (all-MiniLM-L6-v2)
uv run ra "What's the first episode in the results for 'how do I make money with AI?'" --search-type vector_sentence_transformers

# Vector search with higher quality model
uv run ra "What's the first episode in the results for 'how do I make money with AI?'" --search-type vector_sentence_transformers --model all-mpnet-base-v2

# Multilingual vector search
uv run ra "Cuál es el primer episodio que responde a la pregunta 'How do I make money with AI?'" 2000 0.5 --search-type vector_sentence_transformers --model paraphrase-multilingual-MiniLM-L12-v2
```

#### Chunking Strategy Examples
```bash
# Large chunks with high overlap (better context, fewer chunks)
uv run ra "Explain the architecture" 4000 0.8

# Small chunks with low overlap (more precise, more chunks)
uv run ra "Find specific functions" 1000 0.2

# Medium chunks with balanced overlap
uv run ra "What are the main features?" 2000 0.5

# No overlap (fastest processing)
uv run ra "Quick overview" 2000 0.0
```

#### File Type Filtering Examples
```bash
# Only Python files
uv run ra "How does the code work?" --extensions "py"

# Python and Markdown files
uv run ra "What is this project?" --extensions "py,md"

# Multiple file types
uv run ra "Documentation and code" --extensions "py,md,txt,json"

# Only documentation
uv run ra "How to use this?" --extensions "md,mdx"
```

## License

MIT License
