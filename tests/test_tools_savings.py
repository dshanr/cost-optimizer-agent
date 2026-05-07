"""Commitment savings tool tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cost_optimizer.tools.savings import (
    CommitmentSavingsResult,
    calculate_commitment_savings,
)


@pytest.fixture(autouse=True)
def _fixture_pricing_dir(monkeypatch, fixtures_dir: Path):
    monkeypatch.setenv("COST_OPTIMIZER_PRICING_FIXTURES", str(fixtures_dir / "pricing"))


def test_returns_result():
    res = calculate_commitment_savings(
        monthly_on_demand_cost_usd=70.0,
        instance_type="m5.large",
        region="us-east-1",
        term_years=1,
    )
    assert isinstance(res, CommitmentSavingsResult)
    assert res.term_years == 1
    assert res.monthly_savings_usd > 0


def test_3y_better_than_1y_for_same_workload():
    one = calculate_commitment_savings(
        monthly_on_demand_cost_usd=100.0,
        instance_type="t3.xlarge",
        region="us-east-1",
        term_years=1,
    )
    three = calculate_commitment_savings(
        monthly_on_demand_cost_usd=100.0,
        instance_type="t3.xlarge",
        region="us-east-1",
        term_years=3,
    )
    assert three.monthly_savings_usd > one.monthly_savings_usd


def test_savings_uses_pricing_ratios():
    """Given on-demand $0.1664/h, 1y RI $0.1042/h: 1 - 0.1042/0.1664 ~= 37% savings."""
    res = calculate_commitment_savings(
        monthly_on_demand_cost_usd=100.0,
        instance_type="t3.xlarge",
        region="us-east-1",
        term_years=1,
    )
    expected_pct = 1 - (0.1042 / 0.1664)
    assert res.monthly_savings_usd == pytest.approx(100.0 * expected_pct, rel=0.02)


def test_annual_savings_consistent():
    res = calculate_commitment_savings(
        monthly_on_demand_cost_usd=70.0,
        instance_type="m5.large",
        region="us-east-1",
    )
    assert res.annual_savings_usd == pytest.approx(res.monthly_savings_usd * 12, rel=0.001)
