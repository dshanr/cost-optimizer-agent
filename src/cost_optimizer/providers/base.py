"""BillingProvider Protocol: the contract any provider must satisfy."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from cost_optimizer.models import BillingLineItem, ResourceSummary


class BillingProvider(Protocol):
    """Contract for a multi-cloud billing provider.

    A provider knows how to:
    - parse its own CSV format
    - aggregate line items into per-resource summaries
    - expose pricing and utilization tools to the agent
    """

    name: str

    def parse_csv(self, path: Path) -> list[BillingLineItem]: ...
    def aggregate(self, items: list[BillingLineItem]) -> list[ResourceSummary]: ...
    def get_pricing_tool(self) -> Callable[..., object]: ...
    def get_utilization_tool(self) -> Callable[..., object]: ...
