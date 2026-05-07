"""Idle signals tool tests."""
from __future__ import annotations

from cost_optimizer.models import ResourceSummary, UtilizationStats
from cost_optimizer.tools.idle import IdleSignalResult, check_idle_signals


def _resource(rid: str, *, service: str = "EC2", rtype: str | None = "t3.large",
              cost: float = 50.0, hours: float = 720.0,
              utilization: UtilizationStats | None = None) -> ResourceSummary:
    return ResourceSummary(
        resource_id=rid,
        provider="aws",
        service=service,
        resource_type=rtype,
        region="us-east-1",
        monthly_cost_usd=cost,
        usage_hours=hours,
        utilization=utilization,
    )


def test_idle_volume_detected():
    """An EBS volume with no instance association heuristic: vol-* prefix and no rtype."""
    r = _resource("vol-0orphan001", service="EC2", rtype=None)
    res = check_idle_signals(r)
    assert isinstance(res, IdleSignalResult)
    assert res.is_idle is True
    assert "orphan" in res.reasons[0].lower() or "unattached" in res.reasons[0].lower()


def test_idle_low_cpu_with_utilization():
    util = UtilizationStats(
        cpu_p50=1.0, cpu_p95=3.0,
        memory_p50=10.0, memory_p95=15.0,
        network_in_gb_per_day=0.01, network_out_gb_per_day=0.0,
        measurement_window_days=30,
        data_source="mocked",
    )
    r = _resource("i-1", utilization=util)
    res = check_idle_signals(r)
    assert res.is_idle is True
    assert any("cpu" in reason.lower() for reason in res.reasons)


def test_busy_resource_not_idle():
    util = UtilizationStats(
        cpu_p50=70.0, cpu_p95=85.0,
        memory_p50=60.0, memory_p95=75.0,
        network_in_gb_per_day=10.0, network_out_gb_per_day=5.0,
        measurement_window_days=30,
        data_source="mocked",
    )
    r = _resource("i-busy", utilization=util)
    res = check_idle_signals(r)
    assert res.is_idle is False


def test_no_utilization_returns_unknown():
    r = _resource("i-noutil")
    res = check_idle_signals(r)
    assert res.is_idle is False
    assert res.confidence < 0.5
