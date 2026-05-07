"""Aggregator tests."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cost_optimizer.ingest.aggregate import aggregate, top_n_by_cost
from cost_optimizer.ingest.aws_cur import parse_aws_cur
from cost_optimizer.models import BillingLineItem


def _item(rid: str | None, cost: float, hours: float = 720, service: str = "EC2",
          rtype: str | None = "t3.large") -> BillingLineItem:
    return BillingLineItem(
        line_item_id=f"li-{rid or 'none'}-{cost}",
        provider="aws",
        service=service,
        resource_id=rid,
        resource_type=rtype,
        region="us-east-1",
        usage_start=datetime(2026, 4, 1, tzinfo=UTC),
        usage_end=datetime(2026, 4, 30, tzinfo=UTC),
        usage_amount=hours,
        usage_unit="Hrs",
        unblended_cost_usd=cost,
    )


def test_aggregate_groups_by_resource_id():
    items = [_item("i-1", 50), _item("i-1", 30), _item("i-2", 40)]
    summaries = aggregate(items)
    by_id = {s.resource_id: s for s in summaries}
    assert by_id["i-1"].monthly_cost_usd == 80
    assert by_id["i-2"].monthly_cost_usd == 40


def test_aggregate_buckets_unattributed():
    items = [_item(None, 5.0), _item(None, 3.0), _item("i-1", 100)]
    summaries = aggregate(items)
    by_id = {s.resource_id: s for s in summaries}
    assert by_id["unattributed"].monthly_cost_usd == 8.0


def test_aggregate_sorts_by_cost_desc():
    items = [_item("i-cheap", 5), _item("i-mid", 50), _item("i-pricey", 500)]
    summaries = aggregate(items)
    assert [s.resource_id for s in summaries] == ["i-pricey", "i-mid", "i-cheap"]


def test_aggregate_sums_usage_hours():
    items = [_item("i-1", 50, hours=720), _item("i-1", 50, hours=360)]
    summaries = aggregate(items)
    assert summaries[0].usage_hours == 1080


def test_top_n_takes_first_n():
    items = [_item(f"i-{i}", 100 - i) for i in range(10)]
    summaries = aggregate(items)
    top = top_n_by_cost(summaries, n=3)
    assert len(top) == 3
    assert top[0].resource_id == "i-0"
    assert top[2].resource_id == "i-2"


def test_top_n_returns_all_when_n_exceeds():
    items = [_item("i-1", 10), _item("i-2", 20)]
    summaries = aggregate(items)
    top = top_n_by_cost(summaries, n=100)
    assert len(top) == 2


def test_aggregate_uses_first_resource_type_seen():
    items = [_item("i-1", 50, rtype="t3.large"), _item("i-1", 30, rtype=None)]
    summaries = aggregate(items)
    assert summaries[0].resource_type == "t3.large"


def test_aggregate_end_to_end_with_sample(fixtures_dir: Path):
    items = parse_aws_cur(fixtures_dir / "csv" / "sample_aws_cur.csv")
    summaries = aggregate(items)
    rids = {s.resource_id for s in summaries}
    assert "i-0abc111" in rids
    assert "i-0abc222" in rids  # multiple rows summed
    assert "unattributed" in rids  # li-006 has no resource_id

    abc222 = next(s for s in summaries if s.resource_id == "i-0abc222")
    assert abs(abc222.monthly_cost_usd - (69.12 + 34.56)) < 0.01
