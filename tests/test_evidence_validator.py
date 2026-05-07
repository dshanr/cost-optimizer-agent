"""Evidence validator tests."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from cost_optimizer.evidence_validator import validate_recommendation
from cost_optimizer.models import Evidence, Recommendation, RecommendationType


def _rec(reasoning: str, evidence: list[Evidence] | None = None) -> Recommendation:
    return Recommendation(
        recommendation_id=str(uuid4()),
        type=RecommendationType.RIGHTSIZE,
        resource_id="i-1",
        resource_type="t3.xlarge",
        region="us-east-1",
        current_state={"instance_type": "t3.xlarge"},
        recommended_state={"instance_type": "t3.medium"},
        monthly_savings_usd=10.0,
        annual_savings_usd=120.0,
        confidence=0.8,
        effort="low",
        risk_level="medium",
        reasoning=reasoning,
        evidence=evidence or [],
        prerequisites=[],
        rollback_plan=None,
        generated_at=datetime.now(UTC),
        agent_version="0.1.0",
        trace_id=None,
    )


def test_supported_percentage_passes():
    rec = _rec(
        "Instance shows CPU p95 of 14% over 30 days.",
        [Evidence(description="cpu", source="utilization", data={"value": 14.0})],
    )
    ok, missing = validate_recommendation(rec)
    assert ok
    assert missing == []


def test_unsupported_percentage_fails():
    rec = _rec(
        "Instance shows CPU p95 of 14% over 30 days.",
        [Evidence(description="cpu", source="utilization", data={"value": 99.0})],
    )
    ok, missing = validate_recommendation(rec)
    assert not ok
    assert any("14" in m for m in missing)


def test_supported_dollar_amount_passes():
    rec = _rec(
        "Saves $90.88 per month.",
        [Evidence(description="cost", source="billing", data={"monthly_savings_usd": 90.88})],
    )
    ok, _ = validate_recommendation(rec)
    assert ok


def test_supported_instance_type_passes():
    rec = _rec(
        "Recommend rightsizing to t3.medium for cost reduction.",
        [Evidence(description="target", source="pricing_api",
                  data={"instance_type": "t3.medium"})],
    )
    ok, _ = validate_recommendation(rec)
    assert ok


def test_unsupported_instance_type_fails():
    rec = _rec(
        "Recommend rightsizing to t3.medium for cost reduction.",
        [Evidence(description="target", source="pricing_api",
                  data={"instance_type": "t3.large"})],
    )
    ok, missing = validate_recommendation(rec)
    assert not ok
    assert any("t3.medium" in m for m in missing)


def test_no_numeric_claims_passes():
    rec = _rec(
        "Instance is underutilized; consider downsizing.",
        [],
    )
    ok, _ = validate_recommendation(rec)
    assert ok


def test_tolerance_5_percent_for_dollars():
    rec = _rec(
        "Saves $100 per month.",
        [Evidence(description="cost", source="billing", data={"monthly_savings_usd": 102.0})],
    )
    ok, _ = validate_recommendation(rec)
    assert ok  # within 5%


def test_dollar_outside_tolerance_fails():
    rec = _rec(
        "Saves $100 per month.",
        [Evidence(description="cost", source="billing", data={"monthly_savings_usd": 200.0})],
    )
    ok, missing = validate_recommendation(rec)
    assert not ok
