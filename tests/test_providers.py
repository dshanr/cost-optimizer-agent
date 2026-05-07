"""Provider abstraction contract tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cost_optimizer.providers.aws import AwsProvider
from cost_optimizer.providers.base import BillingProvider
from cost_optimizer.providers.oci import OciProvider


def test_aws_provider_implements_protocol():
    p: BillingProvider = AwsProvider()
    assert p.name == "aws"


def test_aws_provider_parses_csv(fixtures_dir: Path):
    p = AwsProvider()
    items = p.parse_csv(fixtures_dir / "csv" / "sample_aws_cur.csv")
    assert len(items) > 0
    assert all(i.provider == "aws" for i in items)


def test_aws_provider_aggregates(fixtures_dir: Path):
    p = AwsProvider()
    items = p.parse_csv(fixtures_dir / "csv" / "sample_aws_cur.csv")
    summaries = p.aggregate(items)
    assert len(summaries) > 0


def test_aws_provider_get_pricing_tool_callable():
    p = AwsProvider()
    tool = p.get_pricing_tool()
    assert callable(tool)


def test_aws_provider_get_utilization_tool_callable():
    p = AwsProvider()
    tool = p.get_utilization_tool()
    assert callable(tool)


@pytest.mark.xfail(strict=True, reason="OCI provider is a v2 deliverable")
def test_oci_provider_implements_protocol(fixtures_dir: Path):
    p = OciProvider()
    items = p.parse_csv(fixtures_dir / "csv" / "sample_oci_billing.csv")
    assert len(items) > 0
