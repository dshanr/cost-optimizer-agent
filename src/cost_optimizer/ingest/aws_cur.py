"""AWS Cost & Usage Report (CUR) CSV parser."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from cost_optimizer.models import BillingLineItem


REQUIRED_COLUMNS = (
    "identity/LineItemId",
    "lineItem/UsageStartDate",
    "lineItem/UsageEndDate",
    "lineItem/ProductCode",
    "lineItem/ResourceId",
    "lineItem/UsageType",
    "lineItem/UsageAmount",
    "lineItem/UnblendedCost",
    "product/region",
)


SERVICE_MAP = {
    "AmazonEC2": "EC2",
    "AmazonS3": "S3",
    "AmazonRDS": "RDS",
    "AmazonELB": "ELB",
    "AmazonElasticLoadBalancingV2": "ELB",
}


class IngestError(ValueError):
    """Raised when the CSV cannot be parsed."""


def parse_aws_cur(path: Path) -> list[BillingLineItem]:
    """Parse an AWS CUR CSV into a list of BillingLineItem.

    Raises IngestError on missing columns or malformed rows.
    """
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise IngestError(f"{path}: empty file")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise IngestError(f"{path}: missing column(s) {missing}")

        items: list[BillingLineItem] = []
        for line_no, row in enumerate(reader, start=2):  # 1 is header
            try:
                items.append(_row_to_item(row))
            except (ValueError, ValidationError) as e:
                raise IngestError(f"{path}: line {line_no}: {e}") from e
        return items


def _row_to_item(row: dict[str, str]) -> BillingLineItem:
    product_code = row["lineItem/ProductCode"]
    service = SERVICE_MAP.get(product_code, product_code.removeprefix("Amazon"))

    resource_id = row.get("lineItem/ResourceId") or None
    resource_type = row.get("product/instanceType") or None

    usage_unit = _infer_usage_unit(row["lineItem/UsageType"])

    tags = {}
    for key, val in row.items():
        if key.startswith("resourceTags/user:") and val:
            tags[key.removeprefix("resourceTags/user:")] = val

    return BillingLineItem(
        line_item_id=row["identity/LineItemId"],
        provider="aws",
        service=service,
        resource_id=resource_id,
        resource_type=resource_type,
        region=row["product/region"],
        usage_start=_parse_datetime(row["lineItem/UsageStartDate"]),
        usage_end=_parse_datetime(row["lineItem/UsageEndDate"]),
        usage_amount=float(row["lineItem/UsageAmount"]),
        usage_unit=usage_unit,
        unblended_cost_usd=float(row["lineItem/UnblendedCost"]),
        tags=tags,
    )


def _parse_datetime(s: str) -> datetime:
    # AWS CUR uses ISO 8601 with trailing Z
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _infer_usage_unit(usage_type: str) -> str:
    if usage_type.startswith("BoxUsage"):
        return "Hrs"
    if "ByteHrs" in usage_type:
        return "GB-Mo"
    if usage_type.startswith("EBS:VolumeUsage"):
        return "GB-Mo"
    if "DataTransfer" in usage_type:
        return "GB"
    if "Requests" in usage_type:
        return "Requests"
    return "Units"
