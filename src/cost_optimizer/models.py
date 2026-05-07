"""Pydantic v2 models for the cost optimizer agent."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Provider = Literal["aws", "oci", "gcp", "azure"]


class BillingLineItem(BaseModel):
    """Provider-agnostic billing line item."""
    model_config = ConfigDict(extra="forbid")

    line_item_id: str
    provider: Provider
    service: str
    resource_id: str | None = None
    resource_type: str | None = None
    region: str
    usage_start: datetime
    usage_end: datetime
    usage_amount: float
    usage_unit: str
    unblended_cost_usd: float
    tags: dict[str, str] = Field(default_factory=dict)


class UtilizationStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_p50: float | None
    cpu_p95: float | None
    memory_p50: float | None
    memory_p95: float | None
    network_in_gb_per_day: float | None
    network_out_gb_per_day: float | None
    measurement_window_days: int
    data_source: Literal["cloudwatch", "oci_monitoring", "mocked"]


class ResourceSummary(BaseModel):
    """Aggregated per-resource view passed to the agent."""
    model_config = ConfigDict(extra="forbid")

    resource_id: str
    provider: Provider
    service: str
    resource_type: str | None
    region: str
    monthly_cost_usd: float
    usage_hours: float
    utilization: UtilizationStats | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class RecommendationType(str, Enum):  # noqa: UP042
    RIGHTSIZE = "rightsize"
    TERMINATE_IDLE = "terminate_idle"
    PURCHASE_COMMITMENT = "purchase_commitment"
    STORAGE_TIER_TRANSITION = "storage_tier_transition"
    DELETE_ORPHANED = "delete_orphaned"
    SCHEDULE_SHUTDOWN = "schedule_shutdown"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    source: Literal["billing", "utilization", "pricing_api", "rightsizing_catalog"]
    data: dict[str, Any]


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str
    type: RecommendationType
    resource_id: str
    resource_type: str | None
    region: str

    current_state: dict[str, Any]
    recommended_state: dict[str, Any]

    monthly_savings_usd: float
    annual_savings_usd: float
    confidence: float = Field(ge=0.0, le=1.0)
    effort: Literal["low", "medium", "high"]
    risk_level: Literal["low", "medium", "high"]

    reasoning: str
    evidence: list[Evidence]
    prerequisites: list[str] = Field(default_factory=list)
    rollback_plan: str | None = None

    generated_at: datetime
    agent_version: str
    trace_id: str | None = None

    @field_validator("recommendation_id")
    @classmethod
    def _uuid4(cls, v: str) -> str:
        parsed = UUID(v)  # raises ValueError if not a valid UUID at all
        if parsed.version != 4:
            raise ValueError(f"recommendation_id must be a UUID4, got UUID{parsed.version}: {v}")
        return v

    @model_validator(mode="after")
    def _savings_consistency(self) -> Recommendation:
        expected = self.monthly_savings_usd * 12
        if abs(self.annual_savings_usd - expected) > 1.0:
            raise ValueError(
                f"annual_savings_usd ({self.annual_savings_usd}) must be within "
                f"$1 of 12 * monthly_savings_usd ({expected})"
            )
        return self


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    output: dict[str, Any] | str
    is_error: bool = False


class LLMResponse(BaseModel):
    """A single LLM completion: either tool calls to execute or final recommendations."""
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[ToolCall] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    finish_reason: Literal["tool_use", "stop"] = "stop"
    raw_text: str | None = None


class Message(BaseModel):
    """One message in the agent's conversation history."""
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
