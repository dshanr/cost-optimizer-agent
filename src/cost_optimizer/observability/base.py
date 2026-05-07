"""Tracer Protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TraceHandle:
    trace_id: str
    resource_id: str
    tool_calls: int = 0
    llm_calls: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class Tracer(Protocol):
    def start_trace(self, *, resource_id: str) -> TraceHandle: ...
    def record_llm_call(
        self,
        handle: TraceHandle,
        *,
        prompt: str,
        response: str,
        tokens: int,
        latency_ms: float,
    ) -> None: ...
    def record_tool_call(
        self,
        handle: TraceHandle,
        *,
        tool: str,
        input: dict[str, Any],
        output: Any,
        latency_ms: float,
    ) -> None: ...
    def end_trace(
        self,
        handle: TraceHandle,
        *,
        recommendations: list[Any],
        cost_usd: float,
    ) -> None: ...
