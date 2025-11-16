# CLI for Wikipedia Agent

import asyncio
import traceback
from collections.abc import Coroutine
from typing import Any

import typer

from config import DEFAULT_SEARCH_MODE, SearchMode
from evals.evaluate import evaluate_agent
from evals.judge import evaluate_answer
from wikiagent.models import JudgeResult, WikipediaAgentResponse
from wikiagent.wikipagent import query_wikipedia

app = typer.Typer()


def _display_agent_result(
    result: WikipediaAgentResponse, question: str, verbose: bool
) -> None:
    """Display agent result in a consistent format"""
    typer.echo(f"\n❓ Question: {question}")

    if result.error:
        typer.echo(f"❌ Error: {result.error.message}")
        typer.echo(f"💡 Suggestion: {result.error.suggestion}")
        if result.error.technical_details:
            typer.echo(f"🔍 Technical: {result.error.technical_details}")
        return

    if not result.answer:
        typer.echo("❌ No answer available")
        return

    typer.echo(f"💡 Answer: {result.answer.answer}")
    typer.echo(f"🎯 Confidence: {result.answer.confidence:.2f}")
    typer.echo(f"🔍 Tool Calls: {len(result.tool_calls)}")

    if verbose:
        typer.echo("\n📋 Tool Call History:")
        for i, call in enumerate(result.tool_calls, 1):
            typer.echo(f"  {i}. {call['tool_name']}: {call['args']}")

    if result.answer.sources_used:
        typer.echo("\n📚 Sources:")
        for i, source in enumerate(result.answer.sources_used[:10], 1):
            typer.echo(f"  {i}. {source}")

    if verbose and result.answer.reasoning:
        typer.echo(f"\n💭 Reasoning: {result.answer.reasoning}")

    if result.usage:
        typer.echo("\n📊 Token Usage:")
        typer.echo(f"  Input tokens: {result.usage.input_tokens}")
        typer.echo(f"  Output tokens: {result.usage.output_tokens}")
        typer.echo(f"  Total tokens: {result.usage.total_tokens}")


def _display_judge_result(result: JudgeResult, verbose: bool) -> None:
    """Display judge evaluation results"""
    typer.echo("\n📊 Judge Evaluation Results:")
    typer.echo(f"  Overall Score: {result.evaluation.overall_score:.2f}")
    typer.echo(f"  Accuracy: {result.evaluation.accuracy:.2f}")
    typer.echo(f"  Completeness: {result.evaluation.completeness:.2f}")
    typer.echo(f"  Relevance: {result.evaluation.relevance:.2f}")

    if verbose:
        typer.echo(f"\n💭 Judge Reasoning: {result.evaluation.reasoning}")
        typer.echo("\n📊 Judge Token Usage:")
        typer.echo(f"  Input tokens: {result.usage.input_tokens}")
        typer.echo(f"  Output tokens: {result.usage.output_tokens}")
        typer.echo(f"  Total tokens: {result.usage.total_tokens}")


def _handle_error(e: Exception, verbose: bool, context: str = "") -> None:
    """Centralized error handling with verbose traceback"""
    typer.echo(f"❌ Error{context}: {str(e)}", err=True)
    if verbose:
        typer.echo(traceback.format_exc(), err=True)


def _run_async(coro: Coroutine[Any, Any, None]) -> None:
    """Helper to run async functions with error handling"""
    try:
        asyncio.run(coro)
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def wikipedia_ask(
    question: str = typer.Argument(..., help="Question to ask the Wikipedia agent"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool calls"),
    mode: str = typer.Option(
        "evaluation",
        "--mode",
        "-m",
        help="Search mode: evaluation (strict minimums), production (adaptive), or research (comprehensive)",
    ),
):
    # Convert mode string to SearchMode enum
    try:
        search_mode = SearchMode(mode.lower())
    except ValueError:
        typer.echo(
            f"❌ Invalid mode: {mode}. Must be one of: evaluation, production, research",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("🤖 Wikipedia Agent is processing your question...")

    async def run_query():
        try:
            result = await query_wikipedia(question, search_mode=search_mode)
        except Exception as e:
            _handle_error(e, verbose, " during agent query")
            raise

        _display_agent_result(result, question, verbose)

    _run_async(run_query())


@app.command()
def judge(
    question: str = typer.Argument(..., help="Question to evaluate"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed evaluation"
    ),
):
    """Evaluate an answer from the Wikipedia agent using LLM-as-a-Judge"""
    typer.echo("🤖 Wikipedia Agent is processing your question...")

    async def run_evaluation():
        try:
            # First, get answer from agent (use evaluation mode for judge)
            agent_result = await query_wikipedia(
                question, search_mode=SearchMode.EVALUATION
            )
        except Exception as e:
            _handle_error(e, verbose, " during agent query")
            raise

        _display_agent_result(agent_result, question, verbose=False)

        # Now evaluate with judge
        try:
            typer.echo("\n⚖️  Judge is evaluating the answer...")
            judge_result = await evaluate_answer(
                question,
                agent_result.answer,
                tool_calls=agent_result.tool_calls,  # Pass tool calls for context
            )
        except Exception as e:
            _handle_error(e, verbose, " during judge evaluation")
            raise

        _display_judge_result(judge_result, verbose)

    _run_async(run_evaluation())


@app.command()
def evaluate(
    ground_truth: str = typer.Option(
        "evals/ground_truth.json",
        "--ground-truth",
        "-g",
        help="Path to ground truth JSON file",
    ),
    output: str = typer.Option(
        "evals/results/evaluation.json",
        "--output",
        "-o",
        help="Path to output JSON file",
    ),
    mode: str = typer.Option(
        "evaluation",
        "--mode",
        "-m",
        help="Search mode: evaluation (strict minimums), production (adaptive), or research (comprehensive)",
    ),
    judge_model: str = typer.Option(
        None,
        "--judge-model",
        "-j",
        help="Model to use for judging (default: from config)",
    ),
):
    """Run full evaluation workflow on Wikipedia agent"""
    # Convert mode string to SearchMode enum
    try:
        search_mode = SearchMode(mode.lower())
    except ValueError:
        typer.echo(
            f"❌ Invalid mode: {mode}. Must be one of: evaluation, production, research",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("🚀 Starting evaluation workflow...")
    typer.echo(f"  Ground truth: {ground_truth}")
    typer.echo(f"  Output: {output}")
    typer.echo(f"  Search mode: {search_mode}")
    if judge_model:
        typer.echo(f"  Judge model: {judge_model}")

    async def run_evaluation():
        try:
            result_path = await evaluate_agent(
                ground_truth_path=ground_truth,
                output_path=output,
                search_mode=search_mode,
                judge_model=judge_model,
            )
            typer.echo(f"\n✅ Evaluation complete! Results saved to: {result_path}")
        except Exception as e:
            _handle_error(e, verbose=False, context=" during evaluation")
            raise

    _run_async(run_evaluation())


if __name__ == "__main__":
    app()
