"""Heuristic idle / orphaned resource detector."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from cost_optimizer.models import ResourceSummary

# Thresholds
IDLE_CPU_P95 = 5.0
IDLE_NETWORK_GB_PER_DAY = 0.1


class IdleSignalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_idle: bool
    confidence: float
    reasons: list[str]


def check_idle_signals(resource: ResourceSummary) -> IdleSignalResult:
    """Return whether the resource appears idle/orphaned, with reasons."""
    reasons: list[str] = []
    confidence = 0.0

    # Heuristic: orphan EBS volume (resource_id starts with vol- and no instance type)
    if resource.resource_id.startswith("vol-") and resource.resource_type is None:
        reasons.append("Unattached EBS volume (no parent instance)")
        confidence = 0.85

    util = resource.utilization
    if util is not None:
        cpu_low = util.cpu_p95 is not None and util.cpu_p95 < IDLE_CPU_P95
        net_low = (
            util.network_in_gb_per_day is not None
            and util.network_out_gb_per_day is not None
            and util.network_in_gb_per_day < IDLE_NETWORK_GB_PER_DAY
            and util.network_out_gb_per_day < IDLE_NETWORK_GB_PER_DAY
        )
        if cpu_low and net_low:
            reasons.append(f"CPU p95 {util.cpu_p95}% and network <{IDLE_NETWORK_GB_PER_DAY} GB/day")
            confidence = max(confidence, 0.8)
        elif cpu_low:
            reasons.append(f"CPU p95 {util.cpu_p95}% over {util.measurement_window_days}d")
            confidence = max(confidence, 0.6)
    else:
        reasons.append("No utilization data available")
        confidence = max(confidence, 0.3)

    is_idle = confidence >= 0.5 and bool(reasons)
    return IdleSignalResult(
        is_idle=is_idle,
        confidence=round(confidence, 2),
        reasons=reasons,
    )
