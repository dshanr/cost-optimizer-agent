"""Aggregate BillingLineItems into per-resource summaries."""
from __future__ import annotations

from collections import defaultdict

from cost_optimizer.models import BillingLineItem, ResourceSummary

UNATTRIBUTED = "unattributed"


def aggregate(items: list[BillingLineItem]) -> list[ResourceSummary]:
    """Group line items by resource and produce per-resource summaries.

    Rows with no resource_id are bucketed into a synthetic 'unattributed' summary
    that the agent should not analyze.
    """
    grouped: dict[tuple[str, str], list[BillingLineItem]] = defaultdict(list)
    for item in items:
        rid = item.resource_id or UNATTRIBUTED
        grouped[(item.provider, rid)].append(item)

    summaries: list[ResourceSummary] = []
    for (provider, rid), group in grouped.items():
        cost = sum(i.unblended_cost_usd for i in group)
        hours = sum(i.usage_amount for i in group if i.usage_unit == "Hrs")
        first = group[0]
        rtype = next((i.resource_type for i in group if i.resource_type), None)
        service = first.service
        region = first.region
        tags: dict[str, str] = {}
        for i in group:
            tags.update(i.tags)
        summaries.append(
            ResourceSummary(
                resource_id=rid,
                provider=provider,  # type: ignore[arg-type]
                service=service,
                resource_type=rtype,
                region=region,
                monthly_cost_usd=round(cost, 2),
                usage_hours=hours,
                utilization=None,
                tags=tags,
            )
        )

    summaries.sort(key=lambda s: s.monthly_cost_usd, reverse=True)
    return summaries


def top_n_by_cost(summaries: list[ResourceSummary], n: int) -> list[ResourceSummary]:
    """Return the top-N most expensive resources, excluding 'unattributed'."""
    candidates = [s for s in summaries if s.resource_id != UNATTRIBUTED]
    return candidates[:n]
