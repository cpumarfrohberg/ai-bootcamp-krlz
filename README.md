# Research Assistant

A RAG (Retrieval-Augmented Generation) powered research assistant for document analysis and question answering.

## Features

- **Document Processing**: Chunk and process various document formats
- **GitHub Integration**: Parse and analyze GitHub repositories
- **RAG Pipeline**: Advanced retrieval and generation capabilities
- **CLI Interface**: Easy-to-use command-line interface
- **Web API**: FastAPI-based REST API for programmatic access

## Prerequisites

- Python 3.11+
- Git
- uv (recommended) or pip

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd research_assistant

# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```bash
# Run the CLI
uv run ra --help

# Process documents
uv run ra process --help

# Start the web API
uv run ra serve
```

## Project Structure

```
research_assistant/
├── cli.py         # Command-line interface
├── config.py      # Configuration settings
├── chunking/      # Document chunking utilities
├── core/          # Core RAG functionality
├── github/        # GitHub integration
├── rag/           # RAG utilities
└── tests/         # Test suite
```

## Configuration

The application uses environment variables for configuration. Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key

# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Redis
REDIS_URL=redis://localhost:6379
```

## License

MIT License
