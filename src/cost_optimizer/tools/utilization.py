"""Mocked, deterministic utilization tool.

Returns synthetic utilization keyed off the hash of resource_id, with three
intentional patterns:
- IDs containing 'idle' are clearly idle (cpu_p95 < 10%)
- IDs containing 'hot' are clearly hot (cpu_p95 > 75%)
- Everything else falls in a middle range based on hash
"""
from __future__ import annotations

import hashlib

from cost_optimizer.models import UtilizationStats


def get_utilization_stats(
    resource_id: str,
    provider: str,
    days: int = 30,
) -> UtilizationStats:
    """Return mocked CPU/memory/network utilization for a resource."""
    if "idle" in resource_id.lower():
        return _idle(days)
    if "hot" in resource_id.lower():
        return _hot(days)
    return _from_hash(resource_id, days)


def _idle(days: int) -> UtilizationStats:
    return UtilizationStats(
        cpu_p50=2.0, cpu_p95=4.0,
        memory_p50=15.0, memory_p95=22.0,
        network_in_gb_per_day=0.05, network_out_gb_per_day=0.02,
        measurement_window_days=days,
        data_source="mocked",
    )


def _hot(days: int) -> UtilizationStats:
    return UtilizationStats(
        cpu_p50=72.0, cpu_p95=88.0,
        memory_p50=70.0, memory_p95=85.0,
        network_in_gb_per_day=12.0, network_out_gb_per_day=8.0,
        measurement_window_days=days,
        data_source="mocked",
    )


def _from_hash(resource_id: str, days: int) -> UtilizationStats:
    digest = hashlib.sha256(resource_id.encode()).digest()
    cpu_p50 = (digest[0] / 255) * 60  # 0-60
    cpu_p95 = min(100.0, cpu_p50 + (digest[1] / 255) * 40)
    mem_p50 = (digest[2] / 255) * 70
    mem_p95 = min(100.0, mem_p50 + (digest[3] / 255) * 30)
    return UtilizationStats(
        cpu_p50=round(cpu_p50, 1), cpu_p95=round(cpu_p95, 1),
        memory_p50=round(mem_p50, 1), memory_p95=round(mem_p95, 1),
        network_in_gb_per_day=round((digest[4] / 255) * 5, 2),
        network_out_gb_per_day=round((digest[5] / 255) * 3, 2),
        measurement_window_days=days,
        data_source="mocked",
    )
