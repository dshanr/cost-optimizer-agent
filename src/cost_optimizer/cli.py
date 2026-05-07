"""CLI entry point."""
from __future__ import annotations

import os
from pathlib import Path

import typer

from cost_optimizer.llm.mock import MockLLM
from cost_optimizer.providers.aws import AwsProvider
from cost_optimizer.runner import Runner

app = typer.Typer(help="Cloud Cost Optimizer Agent")


@app.callback()
def _main() -> None:
    """Cloud Cost Optimizer Agent."""


@app.command()
def run(
    csv_path: Path = typer.Argument(..., exists=True, readable=True),  # noqa: B008
    top_n: int = typer.Option(50, "--top-n", min=1),  # noqa: B008
    llm: str = typer.Option("mock", "--llm", help="mock | claude"),  # noqa: B008
) -> None:
    """Run the agent against a billing CSV and emit recommendations as JSONL."""
    llm_impl = _build_llm(llm)
    runner = Runner(provider=AwsProvider(), llm=llm_impl)
    result = runner.run(csv_path, top_n=top_n)

    for rec in result.recommendations:
        typer.echo(rec.model_dump_json())

    typer.echo(
        f"# analyzed={result.analyzed_count} failed={result.failed_count} "
        f"recommendations={len(result.recommendations)}",
        err=True,
    )


def _build_llm(name: str):
    if name == "mock":
        return MockLLM()
    if name == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            typer.echo("ANTHROPIC_API_KEY not set", err=True)
            raise typer.Exit(code=2)
        from cost_optimizer.llm.claude import ClaudeLLM
        return ClaudeLLM()
    typer.echo(f"unknown llm: {name}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
