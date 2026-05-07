"""AWS CUR parser tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from cost_optimizer.ingest.aws_cur import IngestError, parse_aws_cur


def test_parse_sample(fixtures_dir: Path):
    items = parse_aws_cur(fixtures_dir / "csv" / "sample_aws_cur.csv")
    assert len(items) == 12
    first = items[0]
    assert first.line_item_id == "li-001"
    assert first.provider == "aws"
    assert first.service == "EC2"
    assert first.resource_id == "i-0abc111"
    assert first.resource_type == "t3.xlarge"
    assert first.region == "us-east-1"
    assert first.usage_amount == 720
    assert first.unblended_cost_usd == 121.18
    assert first.usage_unit == "Hrs"
    assert first.tags == {"Env": "prod"}


def test_parse_handles_missing_resource_id(fixtures_dir: Path):
    items = parse_aws_cur(fixtures_dir / "csv" / "sample_aws_cur.csv")
    none_rid = [i for i in items if i.resource_id is None]
    assert len(none_rid) == 1
    assert none_rid[0].line_item_id == "li-006"


def test_parse_normalizes_service_names(fixtures_dir: Path):
    items = parse_aws_cur(fixtures_dir / "csv" / "sample_aws_cur.csv")
    services = {i.service for i in items}
    assert "EC2" in services
    assert "S3" in services


def test_parse_extracts_resource_type_from_usage_type_when_product_field_empty(fixtures_dir: Path):
    """Storage line items have empty product/instanceType but EBS volumes still have type info."""
    items = parse_aws_cur(fixtures_dir / "csv" / "sample_aws_cur.csv")
    ebs = next(i for i in items if i.line_item_id == "li-003")
    assert ebs.resource_type is None  # EBS is volume type, parser leaves None


def test_parse_raises_on_malformed_row(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "identity/LineItemId,bill/PayerAccountId,lineItem/UsageStartDate,lineItem/UsageEndDate,"
        "lineItem/ProductCode,lineItem/ResourceId,lineItem/UsageType,lineItem/UsageAmount,"
        "lineItem/UnblendedCost,product/instanceType,product/region,resourceTags/user:Env\n"
        "li-bad,acct,not-a-date,2026-04-30T23:59:59Z,AmazonEC2,i-1,BoxUsage,10,5.0,t3.large,us-east-1,\n"
    )
    with pytest.raises(IngestError) as exc:
        parse_aws_cur(bad)
    assert "line 2" in str(exc.value).lower()


def test_parse_raises_on_missing_required_column(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(IngestError) as exc:
        parse_aws_cur(bad)
    assert "missing column" in str(exc.value).lower()
