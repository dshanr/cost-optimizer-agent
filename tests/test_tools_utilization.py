"""Utilization tool tests."""
from __future__ import annotations

import pytest

from cost_optimizer.models import UtilizationStats
from cost_optimizer.tools.utilization import get_utilization_stats


def test_returns_utilization_stats():
    res = get_utilization_stats("i-0abc111", "aws", days=30)
    assert isinstance(res, UtilizationStats)
    assert res.measurement_window_days == 30
    assert res.data_source == "mocked"


def test_deterministic():
    a = get_utilization_stats("i-0abc111", "aws")
    b = get_utilization_stats("i-0abc111", "aws")
    assert a == b


def test_different_resources_different_results():
    a = get_utilization_stats("i-resource-a", "aws")
    b = get_utilization_stats("i-resource-b", "aws")
    assert a != b


def test_idle_pattern_for_known_idle_id():
    """resources with 'idle' in id deterministically idle (low cpu)."""
    res = get_utilization_stats("i-0idle555", "aws")
    assert res.cpu_p95 is not None
    assert res.cpu_p95 < 10.0


def test_hot_pattern_for_known_hot_id():
    res = get_utilization_stats("i-0hot999", "aws")
    assert res.cpu_p95 is not None
    assert res.cpu_p95 > 75.0


def test_normal_pattern_in_middle_range():
    res = get_utilization_stats("i-0abc111", "aws")
    assert res.cpu_p95 is not None
    assert 0.0 <= res.cpu_p95 <= 100.0


def test_window_days_passed_through():
    res = get_utilization_stats("i-1", "aws", days=7)
    assert res.measurement_window_days == 7
