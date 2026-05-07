"""JsonlTracer: writes one trace summary per JSON line."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cost_optimizer.observability.base import TraceHandle


class JsonlTracer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = self.output_dir / f"trace-{ts}-{uuid4().hex[:6]}.jsonl"

    def start_trace(self, *, resource_id: str) -> TraceHandle:
        return TraceHandle(trace_id=f"trace-{uuid4().hex[:12]}", resource_id=resource_id)

    def record_llm_call(self, handle, *, prompt, response, tokens, latency_ms):
        handle.llm_calls += 1

    def record_tool_call(self, handle, *, tool, input, output, latency_ms):
        handle.tool_calls += 1

    def end_trace(self, handle, *, recommendations, cost_usd):
        payload: dict[str, Any] = {
            "trace_id": handle.trace_id,
            "resource_id": handle.resource_id,
            "tool_calls": handle.tool_calls,
            "llm_calls": handle.llm_calls,
            "recommendations": len(recommendations),
            "cost_usd": cost_usd,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(payload) + "\n")
