# CLI for Wikipedia Agent

import typer

app = typer.Typer()


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
    import asyncio

    from config import SearchMode
    from wikiagent.wikipagent import query_wikipedia

    try:
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
                typer.echo(f"❌ Error during agent query: {str(e)}", err=True)
                if verbose:
                    import traceback

                    typer.echo(traceback.format_exc(), err=True)
                raise

            typer.echo(f"\n❓ Question: {question}")
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

        asyncio.run(run_query())

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def judge(
    question: str = typer.Argument(..., help="Question to evaluate"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed evaluation"
    ),
):
    """Evaluate an answer from the Wikipedia agent using LLM-as-a-Judge"""
    import asyncio

    from config import SearchMode
    from evals.judge import evaluate_answer
    from wikiagent.wikipagent import query_wikipedia

    try:
        typer.echo("🤖 Wikipedia Agent is processing your question...")

        async def run_evaluation():
            try:
                # First, get answer from agent (use evaluation mode for judge)
                agent_result = await query_wikipedia(
                    question, search_mode=SearchMode.EVALUATION
                )
            except Exception as e:
                typer.echo(f"❌ Error during agent query: {str(e)}", err=True)
                if verbose:
                    import traceback

                    typer.echo(traceback.format_exc(), err=True)
                raise

            typer.echo(f"\n❓ Question: {question}")
            typer.echo(f"💡 Answer: {agent_result.answer.answer}")
            typer.echo(f"🎯 Confidence: {agent_result.answer.confidence:.2f}")

            if agent_result.answer.sources_used:
                typer.echo("\n📚 Sources:")
                for i, source in enumerate(agent_result.answer.sources_used[:10], 1):
                    typer.echo(f"  {i}. {source}")

            # Now evaluate with judge
            try:
                typer.echo("\n⚖️  Judge is evaluating the answer...")
                evaluation, judge_usage = await evaluate_answer(
                    question,
                    agent_result.answer,
                    tool_calls=agent_result.tool_calls,  # Pass tool calls for context
                )
            except Exception as e:
                typer.echo(f"❌ Error during judge evaluation: {str(e)}", err=True)
                if verbose:
                    import traceback

                    typer.echo(traceback.format_exc(), err=True)
                raise

            # Display evaluation results
            typer.echo("\n📊 Judge Evaluation Results:")
            typer.echo(f"  Overall Score: {evaluation.overall_score:.2f}")
            typer.echo(f"  Accuracy: {evaluation.accuracy:.2f}")
            typer.echo(f"  Completeness: {evaluation.completeness:.2f}")
            typer.echo(f"  Relevance: {evaluation.relevance:.2f}")

            if verbose:
                typer.echo(f"\n💭 Judge Reasoning: {evaluation.reasoning}")
                typer.echo("\n📊 Judge Token Usage:")
                typer.echo(f"  Input tokens: {judge_usage['input_tokens']}")
                typer.echo(f"  Output tokens: {judge_usage['output_tokens']}")
                typer.echo(f"  Total tokens: {judge_usage['total_tokens']}")

        asyncio.run(run_evaluation())

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
