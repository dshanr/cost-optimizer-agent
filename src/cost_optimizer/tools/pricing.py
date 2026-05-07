"""AWS pricing tool. Fixture-backed by default; live API behind env flag."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AwsPricingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_type: str
    region: str
    operating_system: Literal["Linux", "Windows"]
    tenancy: Literal["Shared", "Dedicated"]
    on_demand_usd_per_hour: float
    ri_1y_no_upfront_usd_per_hour: float
    ri_3y_no_upfront_usd_per_hour: float
    savings_plan_1y_usd_per_hour: float
    savings_plan_3y_usd_per_hour: float


class PricingNotFoundError(LookupError):
    """Raised when no pricing fixture matches the request."""


def get_aws_pricing(
    instance_type: str,
    region: str,
    operating_system: Literal["Linux", "Windows"] = "Linux",
    tenancy: Literal["Shared", "Dedicated"] = "Shared",
) -> AwsPricingResult:
    """Return on-demand and commitment pricing for an EC2 instance type.

    Reads from `tests/fixtures/pricing/aws/<region>/<instance_type>.json` by
    default. Set COST_OPTIMIZER_LIVE_PRICING=1 to hit the live AWS Pricing API
    (not yet implemented).
    """
    if os.environ.get("COST_OPTIMIZER_LIVE_PRICING") == "1":
        raise NotImplementedError(
            "live AWS Pricing API not yet implemented; v1 uses disk fixtures only"
        )

    fixtures_dir = Path(
        os.environ.get("COST_OPTIMIZER_PRICING_FIXTURES")
        or Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "pricing"
    )
    candidate = fixtures_dir / "aws" / region / f"{instance_type}.json"
    if not candidate.exists():
        raise PricingNotFoundError(
            f"no pricing fixture for {instance_type} in {region} at {candidate}"
        )
    payload = json.loads(candidate.read_text())
    payload.setdefault("operating_system", operating_system)
    payload.setdefault("tenancy", tenancy)
    return AwsPricingResult.model_validate(payload)
