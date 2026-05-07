"""Langfuse adapter implementing the Tracer protocol."""
from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from cost_optimizer.observability.base import TraceHandle


class LangfuseTracer:
    def __init__(self) -> None:
        from langfuse import Langfuse  # imported lazily so import succeeds without env
        self._client = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        )
        self._traces: dict[str, Any] = {}

    def start_trace(self, *, resource_id: str) -> TraceHandle:
        trace_id = f"trace-{uuid4().hex[:12]}"
        trace = self._client.trace(
            id=trace_id,
            name="cost_optimizer.agent",
            metadata={"resource_id": resource_id},
        )
        handle = TraceHandle(trace_id=trace_id, resource_id=resource_id)
        self._traces[trace_id] = trace
        return handle

    def record_llm_call(self, handle, *, prompt, response, tokens, latency_ms):
        trace = self._traces.get(handle.trace_id)
        if trace is None:
            return
        trace.generation(
            name="llm_call",
            model="mock-or-claude",
            input=prompt,
            output=response,
            usage={"total_tokens": tokens},
        )
        handle.llm_calls += 1

    def record_tool_call(self, handle, *, tool, input, output, latency_ms):
        trace = self._traces.get(handle.trace_id)
        if trace is None:
            return
        trace.span(name=f"tool:{tool}", input=input, output=output)
        handle.tool_calls += 1

    def end_trace(self, handle, *, recommendations, cost_usd):
        trace = self._traces.pop(handle.trace_id, None)
        if trace is None:
            return
        trace.update(
            output={"recommendations": len(recommendations), "cost_usd": cost_usd}
        )
        self._client.flush()
