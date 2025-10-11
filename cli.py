# CLI for Research Assistant using GitHub repository RAG

import typer

from config import DEFAULT_ALLOWED_EXTENSIONS, DEFAULT_REPO_NAME, DEFAULT_REPO_OWNER
from core.text_rag import TextRAG

app = typer.Typer()

# Global RAG instance
rag = None


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    repo_owner: str = typer.Option(
        DEFAULT_REPO_OWNER, "--owner", "-o", help="GitHub repository owner"
    ),
    repo_name: str = typer.Option(
        DEFAULT_REPO_NAME, "--repo", "-r", help="GitHub repository name"
    ),
    extensions: str = typer.Option(
        ",".join(DEFAULT_ALLOWED_EXTENSIONS),
        "--extensions",
        "-e",
        help="Comma-separated file extensions to include (e.g., md,mdx)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Ask a question and get an answer from GitHub repository using text-based RAG"""

    global rag

    try:
        # Parse extensions
        allowed_extensions = (
            set(extensions.split(",")) if extensions else DEFAULT_ALLOWED_EXTENSIONS
        )

        if verbose:
            typer.echo(f"🔍 Searching for: {question}")
            typer.echo(f"📁 Repository: {repo_owner}/{repo_name}")
            typer.echo(f"📄 Extensions: {allowed_extensions}")

        # Initialize RAG if not already done
        if rag is None:
            if verbose:
                typer.echo("📥 Loading repository data...")

            # Initialize and load repository using TextRAG
            rag = TextRAG()
            rag.load_repository(
                repo_owner=repo_owner,
                repo_name=repo_name,
                allowed_extensions=allowed_extensions,
            )

            if verbose:
                typer.echo(f"📚 Loaded {len(rag.documents)} files")
                typer.echo(f"📝 Created {len(rag.chunks)} document chunks")

        # Get answer from RAG
        answer = rag.ask(question)

        # Get search results for metadata
        search_results = rag.index.search(question, num_results=3)

        typer.echo(f"\n❓ Question: {question}")
        typer.echo(f"💡 Answer: {answer}")
        typer.echo("🔧 Method: text search")

        if verbose and search_results:
            typer.echo(f"\n🔍 Found {len(search_results)} relevant documents")

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
