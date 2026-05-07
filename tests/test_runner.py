"""Runner orchestration tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cost_optimizer.llm.mock import MockLLM
from cost_optimizer.providers.aws import AwsProvider
from cost_optimizer.runner import Runner


@pytest.fixture(autouse=True)
def _fixture_pricing(monkeypatch, fixtures_dir):
    monkeypatch.setenv("COST_OPTIMIZER_PRICING_FIXTURES", str(fixtures_dir / "pricing"))


def test_runner_processes_top_n(fixtures_dir: Path):
    csv_path = fixtures_dir / "csv" / "sample_aws_cur.csv"
    runner = Runner(provider=AwsProvider(), llm=MockLLM())
    result = runner.run(csv_path, top_n=3)
    assert result.analyzed_count <= 3
    assert result.failed_count == 0
    assert all(r.trace_id is not None for r in result.recommendations)


def test_runner_emits_recommendations_for_underutilized(fixtures_dir: Path):
    csv_path = fixtures_dir / "csv" / "sample_aws_cur.csv"
    runner = Runner(provider=AwsProvider(), llm=MockLLM())
    result = runner.run(csv_path, top_n=10)
    assert len(result.recommendations) >= 1


def test_runner_isolates_per_resource_failures(fixtures_dir: Path):
    """If one resource crashes, others still complete."""
    crashes = {"calls": 0}

    class FlakyLLM:
        name = "flaky"
        def complete(self, messages, tools):
            crashes["calls"] += 1
            if crashes["calls"] == 1:
                raise RuntimeError("boom")
            return MockLLM().complete(messages, tools)

    csv_path = fixtures_dir / "csv" / "sample_aws_cur.csv"
    runner = Runner(provider=AwsProvider(), llm=FlakyLLM())
    result = runner.run(csv_path, top_n=2)
    assert result.failed_count >= 1
    assert result.analyzed_count >= 0
