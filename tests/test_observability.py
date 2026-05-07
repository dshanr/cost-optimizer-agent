"""Tracer tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cost_optimizer.agent import Agent
from cost_optimizer.llm.mock import MockLLM
from cost_optimizer.models import ResourceSummary
from cost_optimizer.observability.base import Tracer
from cost_optimizer.observability.jsonl_tracer import JsonlTracer


@pytest.fixture(autouse=True)
def _fixture_pricing(monkeypatch, fixtures_dir):
    monkeypatch.setenv("COST_OPTIMIZER_PRICING_FIXTURES", str(fixtures_dir / "pricing"))


def _resource() -> ResourceSummary:
    return ResourceSummary(
        resource_id="i-under-001", provider="aws", service="EC2",
        resource_type="t3.xlarge", region="us-east-1",
        monthly_cost_usd=121.18, usage_hours=720.0, utilization=None,
    )


def test_jsonl_tracer_writes_file(tmp_path: Path):
    tracer = JsonlTracer(output_dir=tmp_path)
    handle = tracer.start_trace(resource_id="i-1")
    tracer.record_tool_call(handle, tool="get_aws_pricing",
                            input={}, output={"x": 1}, latency_ms=12)
    tracer.record_llm_call(handle, prompt="hi", response="hello",
                           tokens=10, latency_ms=200)
    tracer.end_trace(handle, recommendations=[], cost_usd=0.001)

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["resource_id"] == "i-1"
    assert payload["tool_calls"] == 1
    assert payload["llm_calls"] == 1


def test_agent_uses_tracer(tmp_path: Path):
    tracer = JsonlTracer(output_dir=tmp_path)
    agent = Agent(llm=MockLLM(), tracer=tracer)
    agent.run(_resource())
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1


def test_tracer_is_protocol():
    """JsonlTracer satisfies the Tracer Protocol (structural typing check)."""
    t: Tracer = JsonlTracer(output_dir=Path("/tmp"))
    assert hasattr(t, "start_trace")
