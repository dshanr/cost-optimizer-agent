"""Rightsizing tool tests."""
from __future__ import annotations

import pytest

from cost_optimizer.tools.rightsizing import (
    InstanceOption,
    UnknownInstanceFamilyError,
    get_rightsizing_options,
)


def test_returns_smaller_options_in_same_family():
    opts = get_rightsizing_options("t3.xlarge")
    types = [o.instance_type for o in opts]
    assert "t3.large" in types
    assert "t3.medium" in types
    assert "t3.2xlarge" not in types  # bigger excluded
    assert "t3.xlarge" not in types  # current excluded


def test_returns_pydantic_models():
    opts = get_rightsizing_options("t3.xlarge")
    assert all(isinstance(o, InstanceOption) for o in opts)


def test_smallest_first():
    opts = get_rightsizing_options("t3.xlarge")
    sizes = [o.vcpu for o in opts]
    assert sizes == sorted(sizes)


def test_target_utilization_filters_too_small():
    opts = get_rightsizing_options("t3.xlarge", target_cpu_utilization=0.6)
    assert len(opts) > 0


def test_unknown_family_raises():
    with pytest.raises(UnknownInstanceFamilyError):
        get_rightsizing_options("zz9.plural-z-alpha")


def test_no_options_for_smallest_in_family():
    opts = get_rightsizing_options("t3.nano")
    assert opts == []
