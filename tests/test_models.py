"""Pydantic model contract tests."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cost_optimizer.models import (
    BillingLineItem,
    Evidence,
    Recommendation,
    RecommendationType,
    ResourceSummary,
    UtilizationStats,
)


def _now() -> datetime:
    return datetime(2026, 5, 6, 12, 0, tzinfo=UTC)


def test_billing_line_item_round_trip():
    item = BillingLineItem(
        line_item_id="li-1",
        provider="aws",
        service="EC2",
        resource_id="i-0abc",
        resource_type="t3.large",
        region="us-east-1",
        usage_start=_now(),
        usage_end=_now(),
        usage_amount=24.0,
        usage_unit="Hrs",
        unblended_cost_usd=12.34,
        tags={"Env": "prod"},
    )
    payload = item.model_dump_json()
    restored = BillingLineItem.model_validate_json(payload)
    assert restored == item


def test_billing_line_item_rejects_bad_provider():
    with pytest.raises(ValidationError):
        BillingLineItem(
            line_item_id="li-1",
            provider="digitalocean",  # not allowed
            service="EC2",
            resource_id="i-1",
            resource_type=None,
            region="us-east-1",
            usage_start=_now(),
            usage_end=_now(),
            usage_amount=1.0,
            usage_unit="Hrs",
            unblended_cost_usd=1.0,
        )


def test_resource_summary_no_utilization_ok():
    rs = ResourceSummary(
        resource_id="i-1",
        provider="aws",
        service="EC2",
        resource_type="t3.large",
        region="us-east-1",
        monthly_cost_usd=100.0,
        usage_hours=720.0,
        utilization=None,
    )
    assert rs.utilization is None


def test_utilization_stats_data_source_validated():
    with pytest.raises(ValidationError):
        UtilizationStats(
            cpu_p50=10.0, cpu_p95=20.0,
            memory_p50=None, memory_p95=None,
            network_in_gb_per_day=None, network_out_gb_per_day=None,
            measurement_window_days=30,
            data_source="prometheus",  # not allowed
        )


def test_recommendation_confidence_bounds():
    base = _recommendation_kwargs()
    Recommendation(**base, confidence=0.0)
    Recommendation(**base, confidence=1.0)
    with pytest.raises(ValidationError):
        Recommendation(**base, confidence=1.5)
    with pytest.raises(ValidationError):
        Recommendation(**base, confidence=-0.1)


def test_recommendation_savings_consistency():
    """annual_savings_usd must be ~= 12 * monthly_savings_usd."""
    base = _recommendation_kwargs()
    base["monthly_savings_usd"] = 100.0
    base["annual_savings_usd"] = 1200.0
    Recommendation(**base, confidence=0.8)

    base["annual_savings_usd"] = 100.0  # inconsistent
    with pytest.raises(ValidationError):
        Recommendation(**base, confidence=0.8)


def test_recommendation_id_must_be_uuid4():
    base = _recommendation_kwargs()
    base["recommendation_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        Recommendation(**base, confidence=0.8)


def test_recommendation_id_must_be_v4_specifically():
    """A valid UUID1 must be rejected — only UUID4 is accepted."""
    base = _recommendation_kwargs()
    base["recommendation_id"] = "00000000-0000-1000-8000-000000000000"  # UUID1 shape
    with pytest.raises(ValidationError):
        Recommendation(**base, confidence=0.8)


def test_recommendation_type_enum_values():
    assert RecommendationType.RIGHTSIZE.value == "rightsize"
    assert RecommendationType.TERMINATE_IDLE.value == "terminate_idle"
    assert RecommendationType.PURCHASE_COMMITMENT.value == "purchase_commitment"
    assert RecommendationType.STORAGE_TIER_TRANSITION.value == "storage_tier_transition"
    assert RecommendationType.DELETE_ORPHANED.value == "delete_orphaned"
    assert RecommendationType.SCHEDULE_SHUTDOWN.value == "schedule_shutdown"


def test_evidence_source_enum():
    Evidence(description="cpu", source="utilization", data={"v": 14.2})
    with pytest.raises(ValidationError):
        Evidence(description="cpu", source="oracle", data={})


def _recommendation_kwargs() -> dict:
    return dict(
        recommendation_id=str(uuid4()),
        type=RecommendationType.RIGHTSIZE,
        resource_id="i-1",
        resource_type="t3.xlarge",
        region="us-east-1",
        current_state={"instance_type": "t3.xlarge"},
        recommended_state={"instance_type": "t3.medium"},
        monthly_savings_usd=10.0,
        annual_savings_usd=120.0,
        effort="low",
        risk_level="medium",
        reasoning="Low utilization observed.",
        evidence=[Evidence(description="cpu p95", source="utilization", data={"value": 14.0})],
        prerequisites=[],
        rollback_plan=None,
        generated_at=_now(),
        agent_version="0.1.0",
        trace_id=None,
    )
