"""Eval harness tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cost_optimizer.llm.mock import MockLLM
from evals.eval_runner import EvalReport, run_eval


@pytest.fixture(autouse=True)
def _fixture_pricing(monkeypatch, fixtures_dir):
    monkeypatch.setenv("COST_OPTIMIZER_PRICING_FIXTURES", str(fixtures_dir / "pricing"))


def test_eval_runs_all_cases():
    report = run_eval(llm=MockLLM())
    assert isinstance(report, EvalReport)
    assert report.case_count == 5


def test_eval_meets_thresholds():
    report = run_eval(llm=MockLLM())
    assert report.precision >= 0.85, f"precision {report.precision}"
    assert report.recall >= 0.80, f"recall {report.recall}"


def test_eval_emits_markdown(tmp_path: Path):
    report = run_eval(llm=MockLLM(), output_dir=tmp_path)
    files = list(tmp_path.glob("*.md"))
    assert len(files) >= 1
    body = files[0].read_text()
    assert "Eval Report" in body
    assert "precision" in body.lower()
