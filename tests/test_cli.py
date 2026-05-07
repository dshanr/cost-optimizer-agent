"""CLI tests via typer.testing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cost_optimizer.cli import app


@pytest.fixture(autouse=True)
def _fixture_pricing(monkeypatch, fixtures_dir):
    monkeypatch.setenv("COST_OPTIMIZER_PRICING_FIXTURES", str(fixtures_dir / "pricing"))


def test_cli_run_emits_jsonl(fixtures_dir: Path):
    runner = CliRunner()
    csv = fixtures_dir / "csv" / "sample_aws_cur.csv"
    result = runner.invoke(app, ["run", str(csv), "--top-n", "5", "--llm", "mock"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("{")]
    assert lines, "expected at least one JSON line"
    parsed = json.loads(lines[0])
    assert "recommendation_id" in parsed


def test_cli_run_unknown_llm_errors(fixtures_dir: Path):
    runner = CliRunner()
    csv = fixtures_dir / "csv" / "sample_aws_cur.csv"
    result = runner.invoke(app, ["run", str(csv), "--llm", "bogus"])
    assert result.exit_code != 0


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
