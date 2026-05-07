"""Pricing tool tests (fixture-backed by default)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cost_optimizer.tools.pricing import (
    AwsPricingResult,
    PricingNotFoundError,
    get_aws_pricing,
)


@pytest.fixture(autouse=True)
def _fixture_pricing_dir(monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path):
    monkeypatch.setenv("COST_OPTIMIZER_PRICING_FIXTURES", str(fixtures_dir / "pricing"))
    monkeypatch.delenv("COST_OPTIMIZER_LIVE_PRICING", raising=False)


def test_get_aws_pricing_returns_result():
    res = get_aws_pricing(instance_type="t3.xlarge", region="us-east-1")
    assert isinstance(res, AwsPricingResult)
    assert res.instance_type == "t3.xlarge"
    assert res.on_demand_usd_per_hour == pytest.approx(0.1664)
    assert res.ri_1y_no_upfront_usd_per_hour == pytest.approx(0.1042)


def test_get_aws_pricing_missing_raises():
    with pytest.raises(PricingNotFoundError):
        get_aws_pricing(instance_type="x99.galactic", region="us-east-1")


def test_get_aws_pricing_returns_pydantic_model():
    res = get_aws_pricing(instance_type="t3.medium", region="us-east-1")
    payload = res.model_dump()
    assert payload["on_demand_usd_per_hour"] == pytest.approx(0.0416)


def test_get_aws_pricing_live_mode_marker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COST_OPTIMIZER_LIVE_PRICING", "1")
    monkeypatch.delenv("COST_OPTIMIZER_PRICING_FIXTURES", raising=False)
    with pytest.raises(NotImplementedError, match="live"):
        get_aws_pricing(instance_type="t3.medium", region="us-east-1")
