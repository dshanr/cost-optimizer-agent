"""OCI billing provider stub. v2 deliverable.

Exists so the BillingProvider Protocol is exercised by two implementations
in the type checker, even if only AWS works at runtime.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cost_optimizer.models import BillingLineItem, ResourceSummary


class OciProvider:
    name = "oci"

    def parse_csv(self, path: Path) -> list[BillingLineItem]:
        raise NotImplementedError("OCI ingest is a v2 deliverable")

    def aggregate(self, items: list[BillingLineItem]) -> list[ResourceSummary]:
        raise NotImplementedError("OCI aggregate is a v2 deliverable")

    def get_pricing_tool(self) -> Callable[..., object]:
        raise NotImplementedError("OCI pricing is a v2 deliverable")

    def get_utilization_tool(self) -> Callable[..., object]:
        raise NotImplementedError("OCI utilization is a v2 deliverable")
