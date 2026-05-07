"""AWS billing provider."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from cost_optimizer.ingest.aggregate import aggregate as _aggregate
from cost_optimizer.ingest.aws_cur import parse_aws_cur
from cost_optimizer.models import BillingLineItem, ResourceSummary
from cost_optimizer.tools.pricing import get_aws_pricing
from cost_optimizer.tools.utilization import get_utilization_stats


class AwsProvider:
    name = "aws"

    def parse_csv(self, path: Path) -> list[BillingLineItem]:
        return parse_aws_cur(path)

    def aggregate(self, items: list[BillingLineItem]) -> list[ResourceSummary]:
        return _aggregate(items)

    def get_pricing_tool(self) -> Callable[..., object]:
        return get_aws_pricing

    def get_utilization_tool(self) -> Callable[..., object]:
        return get_utilization_stats
