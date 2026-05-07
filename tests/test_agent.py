"""Agent integration tests against MockLLM and the five golden scenarios."""
from __future__ import annotations

from datetime import UTC

import pytest

from cost_optimizer.agent import Agent
from cost_optimizer.llm.mock import MockLLM
from cost_optimizer.models import (
    LLMResponse,
    Recommendation,
    RecommendationType,
    ResourceSummary,
    ToolCall,
)


def _resource(rid: str = "i-rs", *, rtype: str | None = "t3.xlarge",
              cost: float = 121.18, hours: float = 720.0,
              service: str = "EC2") -> ResourceSummary:
    return ResourceSummary(
        resource_id=rid, provider="aws", service=service, resource_type=rtype,
        region="us-east-1", monthly_cost_usd=cost, usage_hours=hours, utilization=None,
    )


@pytest.fixture(autouse=True)
def _fixture_pricing(monkeypatch, fixtures_dir):
    monkeypatch.setenv("COST_OPTIMIZER_PRICING_FIXTURES", str(fixtures_dir / "pricing"))


def test_agent_rightsize_underutilized_t3_xlarge():
    """Golden case: aws-rightsize-001."""
    agent = Agent(llm=MockLLM())
    r = _resource("i-under-001", rtype="t3.xlarge", cost=121.18)
    recs = agent.run(r)
    assert len(recs) == 1
    assert recs[0].type == RecommendationType.RIGHTSIZE
    assert recs[0].recommended_state["instance_type"] == "t3.medium"


def test_agent_terminate_orphan_volume():
    """Golden case: aws-idle-001."""
    agent = Agent(llm=MockLLM())
    r = _resource("vol-0orphan001", rtype=None, cost=8.00, service="EC2", hours=720)
    recs = agent.run(r)
    assert len(recs) == 1
    assert recs[0].type == RecommendationType.TERMINATE_IDLE


def test_agent_zero_recs_for_well_utilized():
    """Golden case: aws-negative-001."""
    agent = Agent(llm=MockLLM())
    r = _resource("i-busy", rtype="t3.medium", cost=30.30)
    recs = agent.run(r)
    assert recs == []


def test_agent_caps_tool_calls():
    """Even if the LLM keeps requesting tools, we stop after MAX_TOOL_CALLS."""
    agent = Agent(llm=_LoopingLLM(), max_tool_calls=3)
    r = _resource()
    recs = agent.run(r)
    assert recs == []


def test_agent_returns_pydantic_recommendations():
    agent = Agent(llm=MockLLM())
    r = _resource("i-under-001", rtype="t3.xlarge", cost=121.18)
    recs = agent.run(r)
    assert all(isinstance(rec, Recommendation) for rec in recs)


def test_agent_drops_unsupported_recommendations():
    """If a recommendation fails evidence validation twice, it is dropped silently."""
    bad = _BadEvidenceLLM()
    agent = Agent(llm=bad)
    r = _resource()
    recs = agent.run(r)
    assert recs == []


def test_agent_attaches_trace_id():
    agent = Agent(llm=MockLLM(), trace_id_factory=lambda: "trace-fixed-123")
    r = _resource("i-under-001", rtype="t3.xlarge", cost=121.18)
    recs = agent.run(r)
    if recs:
        assert recs[0].trace_id == "trace-fixed-123"


# --- helper LLMs ---

class _LoopingLLM:
    name = "looping"

    def complete(self, messages, tools):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc", name="get_aws_pricing",
                                 arguments={"instance_type": "t3.xlarge", "region": "us-east-1"})],
            finish_reason="tool_use",
        )


class _BadEvidenceLLM:
    """Returns a recommendation whose reasoning has an unsupported claim."""
    name = "bad_evidence"
    _calls = 0

    def complete(self, messages, tools):
        from datetime import datetime
        from uuid import uuid4

        from cost_optimizer.models import Evidence
        self._calls += 1
        rec = Recommendation(
            recommendation_id=str(uuid4()),
            type=RecommendationType.RIGHTSIZE,
            resource_id="i-rs",
            resource_type="t3.xlarge",
            region="us-east-1",
            current_state={}, recommended_state={"instance_type": "t3.medium"},
            monthly_savings_usd=10.0, annual_savings_usd=120.0, confidence=0.9,
            effort="low", risk_level="low",
            reasoning="CPU p95 is 14% — clear win.",  # not in evidence
            evidence=[Evidence(description="other", source="utilization", data={"value": 99.0})],
            prerequisites=[], rollback_plan=None,
            generated_at=datetime.now(UTC), agent_version="0.1.0", trace_id=None,
        )
        return LLMResponse(recommendations=[rec], finish_reason="stop")
