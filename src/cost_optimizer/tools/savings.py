"""Commitment-based savings calculator (Reserved Instances, Savings Plans)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from cost_optimizer.tools.pricing import get_aws_pricing


class CommitmentSavingsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_type: str
    region: str
    term_years: Literal[1, 3]
    payment: Literal["no_upfront", "partial_upfront", "all_upfront"]
    on_demand_monthly_cost_usd: float
    committed_monthly_cost_usd: float
    monthly_savings_usd: float
    annual_savings_usd: float
    savings_percent: float


def calculate_commitment_savings(
    monthly_on_demand_cost_usd: float,
    instance_type: str,
    region: str,
    term_years: Literal[1, 3] = 1,
    payment: Literal["no_upfront", "partial_upfront", "all_upfront"] = "no_upfront",
) -> CommitmentSavingsResult:
    """Compute monthly/annual savings if `monthly_on_demand_cost_usd` were
    converted to a 1y or 3y RI."""
    pricing = get_aws_pricing(instance_type=instance_type, region=region)
    on_demand_rate = pricing.on_demand_usd_per_hour
    if term_years == 1:
        committed_rate = pricing.ri_1y_no_upfront_usd_per_hour
    else:
        committed_rate = pricing.ri_3y_no_upfront_usd_per_hour

    ratio = committed_rate / on_demand_rate
    committed_monthly = monthly_on_demand_cost_usd * ratio
    monthly_savings = monthly_on_demand_cost_usd - committed_monthly
    return CommitmentSavingsResult(
        instance_type=instance_type,
        region=region,
        term_years=term_years,
        payment=payment,
        on_demand_monthly_cost_usd=round(monthly_on_demand_cost_usd, 2),
        committed_monthly_cost_usd=round(committed_monthly, 2),
        monthly_savings_usd=round(monthly_savings, 2),
        annual_savings_usd=round(monthly_savings * 12, 2),
        savings_percent=round((1 - ratio) * 100, 2),
    )
