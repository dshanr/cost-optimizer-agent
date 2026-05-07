"""Deterministic mock LLM used in tests, evals, and the default CLI path.

The mock branches on the user-message ResourceSummary to decide whether to
request tool calls (first turn) or emit final recommendations (after tool
results have been observed in the history).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from cost_optimizer.models import (
    Evidence,
    LLMResponse,
    Message,
    Recommendation,
    RecommendationType,
    ResourceSummary,
    ToolCall,
)

AGENT_VERSION = "0.1.0"


class MockLLM:
    name = "mock"

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        resource = _extract_resource(messages)
        if resource is None:
            return LLMResponse(finish_reason="stop")

        if not _has_tool_results(messages):
            return _first_turn(resource)

        return _final_turn(resource)


def _extract_resource(messages: list[Message]) -> ResourceSummary | None:
    for m in messages:
        if m.role == "user" and m.content:
            try:
                return ResourceSummary.model_validate_json(m.content)
            except Exception:
                continue
    return None


def _has_tool_results(messages: list[Message]) -> bool:
    return any(m.role == "tool" for m in messages)


def _first_turn(r: ResourceSummary) -> LLMResponse:
    """Always request pricing + utilization on the first turn."""
    calls: list[ToolCall] = []
    if r.resource_type and r.service == "EC2":
        calls.append(ToolCall(
            id=f"tc_{uuid4().hex[:8]}",
            name="get_aws_pricing",
            arguments={"instance_type": r.resource_type, "region": r.region},
        ))
    if r.utilization is None:
        calls.append(ToolCall(
            id=f"tc_{uuid4().hex[:8]}",
            name="get_utilization_stats",
            arguments={"resource_id": r.resource_id, "provider": r.provider},
        ))
    if not calls:
        calls.append(ToolCall(
            id=f"tc_{uuid4().hex[:8]}",
            name="check_idle_signals",
            arguments={"resource_id": r.resource_id},
        ))
    return LLMResponse(tool_calls=calls, finish_reason="tool_use")


def _final_turn(r: ResourceSummary) -> LLMResponse:
    """Emit recommendations based on resource shape."""
    recs: list[Recommendation] = []

    util = r.utilization
    cpu_p95 = util.cpu_p95 if util else None

    # Idle: stopped/orphan vol or extremely low CPU
    if r.resource_id.startswith("vol-") and r.resource_type is None:
        recs.append(_terminate_idle_rec(r, reason="orphan_volume"))
    elif cpu_p95 is not None and cpu_p95 < 5.0:
        recs.append(_terminate_idle_rec(r, reason="low_cpu", cpu_p95=cpu_p95))

    # Rightsize: under-utilized t3.xlarge / m5 / etc.
    elif (
        cpu_p95 is not None
        and cpu_p95 < 30.0
        and r.resource_type in {"t3.xlarge", "t3.large", "m5.large", "m5.xlarge"}
    ):
        recs.append(_rightsize_rec(r, cpu_p95))

    # Otherwise: no recommendation
    return LLMResponse(recommendations=recs, finish_reason="stop")


def _rightsize_rec(r: ResourceSummary, cpu_p95: float) -> Recommendation:
    target = {
        "t3.xlarge": ("t3.medium", 0.0416, 0.1664),
        "t3.large": ("t3.small", 0.0208, 0.0832),
        "m5.large": ("t3.medium", 0.0416, 0.0960),
        "m5.xlarge": ("m5.large", 0.0960, 0.1920),
    }[r.resource_type]  # type: ignore[index]
    new_type, new_rate, old_rate = target
    monthly_savings = round(r.monthly_cost_usd * (1 - new_rate / old_rate), 2)
    return Recommendation(
        recommendation_id=str(uuid4()),
        type=RecommendationType.RIGHTSIZE,
        resource_id=r.resource_id,
        resource_type=r.resource_type,
        region=r.region,
        current_state={"instance_type": r.resource_type, "monthly_cost_usd": r.monthly_cost_usd},
        recommended_state={"instance_type": new_type},
        monthly_savings_usd=monthly_savings,
        annual_savings_usd=round(monthly_savings * 12, 2),
        confidence=0.86,
        effort="low",
        risk_level="medium",
        reasoning=(
            f"Instance shows CPU p95 of {cpu_p95}% over 30 days. "
            f"A {new_type} provides sufficient headroom; current rate is "
            f"${old_rate}/hour and target rate is ${new_rate}/hour."
        ),
        evidence=[
            Evidence(
                description="30-day CPU p95",
                source="utilization",
                data={"value": cpu_p95, "unit": "percent"},
            ),
            Evidence(
                description=f"Current on-demand price for {r.resource_type}",
                source="pricing_api",
                data={
                    "instance_type": r.resource_type, "region": r.region,
                    "usd_per_hour": old_rate,
                },
            ),
            Evidence(
                description=f"Target on-demand price for {new_type}",
                source="pricing_api",
                data={"instance_type": new_type, "region": r.region, "usd_per_hour": new_rate},
            ),
        ],
        prerequisites=["Verify application memory ceiling via load testing"],
        rollback_plan=f"Stop instance, change type back to {r.resource_type}, start instance.",
        generated_at=datetime.now(timezone.utc),
        agent_version=AGENT_VERSION,
        trace_id=None,
    )


def _terminate_idle_rec(r: ResourceSummary, *, reason: str,
                        cpu_p95: float | None = None) -> Recommendation:
    if reason == "orphan_volume":
        why = f"Unattached EBS volume {r.resource_id} costs ${r.monthly_cost_usd}/mo with no parent instance."
        evidence = [Evidence(
            description="Orphan volume signal",
            source="billing",
            data={"resource_id": r.resource_id, "monthly_cost_usd": r.monthly_cost_usd},
        )]
        confidence = 0.9
    else:
        why = (
            f"Resource {r.resource_id} shows CPU p95 of {cpu_p95}% over 30 days. "
            f"Costing ${r.monthly_cost_usd}/mo for negligible utilization."
        )
        evidence = [
            Evidence(
                description="30-day CPU p95",
                source="utilization",
                data={"value": cpu_p95, "unit": "percent"},
            ),
            Evidence(
                description="Monthly cost",
                source="billing",
                data={"monthly_cost_usd": r.monthly_cost_usd},
            ),
        ]
        confidence = 0.82
    return Recommendation(
        recommendation_id=str(uuid4()),
        type=RecommendationType.TERMINATE_IDLE,
        resource_id=r.resource_id,
        resource_type=r.resource_type,
        region=r.region,
        current_state={"monthly_cost_usd": r.monthly_cost_usd},
        recommended_state={"action": "terminate"},
        monthly_savings_usd=round(r.monthly_cost_usd, 2),
        annual_savings_usd=round(r.monthly_cost_usd * 12, 2),
        confidence=confidence,
        effort="low",
        risk_level="low",
        reasoning=why,
        evidence=evidence,
        prerequisites=["Confirm resource is not pinned by ops team"],
        rollback_plan="Restore from snapshot if available.",
        generated_at=datetime.now(timezone.utc),
        agent_version=AGENT_VERSION,
        trace_id=None,
    )
