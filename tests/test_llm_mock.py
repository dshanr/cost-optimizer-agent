"""MockLLM tests."""
from __future__ import annotations

from cost_optimizer.llm.mock import MockLLM
from cost_optimizer.models import (
    LLMResponse,
    Message,
    RecommendationType,
    ResourceSummary,
    UtilizationStats,
)


def _resource(rid: str = "i-rs", *, rtype: str = "t3.xlarge", cost: float = 121.18,
              cpu_p95: float | None = 14.0, hours: float = 720.0,
              service: str = "EC2") -> ResourceSummary:
    util = None
    if cpu_p95 is not None:
        util = UtilizationStats(
            cpu_p50=cpu_p95 - 5,
            cpu_p95=cpu_p95,
            memory_p50=20.0,
            memory_p95=25.0,
            network_in_gb_per_day=1.0,
            network_out_gb_per_day=0.5,
            measurement_window_days=30,
            data_source="mocked",
        )
    return ResourceSummary(
        resource_id=rid, provider="aws", service=service, resource_type=rtype,
        region="us-east-1", monthly_cost_usd=cost, usage_hours=hours, utilization=util,
    )


def test_mock_llm_first_turn_calls_pricing_tool():
    llm = MockLLM()
    r = _resource()
    sys_msg = Message(role="system", content="...")
    user_msg = Message(role="user", content=r.model_dump_json())
    resp = llm.complete([sys_msg, user_msg], tools=[])
    assert isinstance(resp, LLMResponse)
    assert resp.finish_reason == "tool_use"
    assert any(tc.name == "get_aws_pricing" for tc in resp.tool_calls)


def test_mock_llm_underutilized_t3_xlarge_emits_rightsize():
    """After tool calls have been answered, MockLLM emits a rightsize rec."""
    llm = MockLLM()
    r = _resource("i-under-001", rtype="t3.xlarge", cost=121.18, cpu_p95=14.0)
    history = _ready_for_recommendations(r)
    resp = llm.complete(history, tools=[])
    assert resp.finish_reason == "stop"
    assert len(resp.recommendations) == 1
    rec = resp.recommendations[0]
    assert rec.type == RecommendationType.RIGHTSIZE
    assert rec.recommended_state["instance_type"] == "t3.medium"


def test_mock_llm_idle_resource_emits_terminate():
    llm = MockLLM()
    r = _resource("i-0idle555", rtype=None, cost=16.0, cpu_p95=3.0, service="EC2")
    history = _ready_for_recommendations(r)
    resp = llm.complete(history, tools=[])
    assert resp.finish_reason == "stop"
    assert len(resp.recommendations) >= 1
    assert resp.recommendations[0].type == RecommendationType.TERMINATE_IDLE


def test_mock_llm_well_utilized_emits_zero_recs():
    llm = MockLLM()
    r = _resource("i-busy", rtype="t3.medium", cost=30.30, cpu_p95=78.0)
    history = _ready_for_recommendations(r)
    resp = llm.complete(history, tools=[])
    assert resp.finish_reason == "stop"
    assert resp.recommendations == []


def test_mock_llm_deterministic_for_same_input():
    llm = MockLLM()
    r = _resource()
    h = _ready_for_recommendations(r)
    a = llm.complete(h, tools=[])
    b = llm.complete(h, tools=[])
    assert len(a.recommendations) == len(b.recommendations)
    if a.recommendations:
        assert a.recommendations[0].type == b.recommendations[0].type
        assert a.recommendations[0].recommended_state == b.recommendations[0].recommended_state


def _ready_for_recommendations(r: ResourceSummary) -> list[Message]:
    """Build a history where MockLLM has 'seen' tool results and should emit final recs."""
    return [
        Message(role="system", content="..."),
        Message(role="user", content=r.model_dump_json()),
        Message(role="assistant", content="calling tools"),
        Message(role="tool", content="tool results provided"),
    ]
