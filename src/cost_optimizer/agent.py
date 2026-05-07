"""Single-resource ReAct agent.

For each resource the agent loops:
  LLM.complete -> tool_calls? execute -> append to history -> loop
                -> recommendations? validate -> retry once -> return
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from uuid import uuid4

from cost_optimizer.evidence_validator import validate_recommendation
from cost_optimizer.llm.base import LLM
from cost_optimizer.models import (
    Message,
    Recommendation,
    ResourceSummary,
    ToolCall,
    ToolResult,
    UtilizationStats,
)
from cost_optimizer.tools.idle import check_idle_signals
from cost_optimizer.tools.pricing import get_aws_pricing
from cost_optimizer.tools.rightsizing import get_rightsizing_options
from cost_optimizer.tools.savings import calculate_commitment_savings
from cost_optimizer.tools.utilization import get_utilization_stats

MAX_TOOL_CALLS = 6
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text()

ToolFn = Callable[..., object]
TOOL_REGISTRY: dict[str, ToolFn] = {
    "get_aws_pricing": get_aws_pricing,
    "get_utilization_stats": get_utilization_stats,
    "get_rightsizing_options": get_rightsizing_options,
    "calculate_commitment_savings": calculate_commitment_savings,
    "check_idle_signals": check_idle_signals,
}


class Agent:
    def __init__(
        self,
        llm: LLM,
        *,
        max_tool_calls: int = MAX_TOOL_CALLS,
        trace_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.llm = llm
        self.max_tool_calls = max_tool_calls
        self._trace_id_factory = trace_id_factory or (lambda: f"trace-{uuid4().hex[:12]}")

    def run(self, resource: ResourceSummary) -> list[Recommendation]:
        trace_id = self._trace_id_factory()
        history: list[Message] = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=resource.model_dump_json()),
        ]
        resource_msg_idx = 1  # index of the user message holding the resource JSON

        tool_calls_made = 0
        retried = False

        while True:
            resp = self.llm.complete(history, tools=[])

            if resp.finish_reason == "tool_use" and resp.tool_calls:
                if tool_calls_made >= self.max_tool_calls:
                    return []
                results = self._execute_tools(resp.tool_calls, resource)
                tool_calls_made += len(resp.tool_calls)
                # Fold utilization tool results back into the resource so subsequent
                # LLM turns see the augmented resource state.
                resource = _augment_resource(resource, resp.tool_calls, results)
                history[resource_msg_idx] = Message(
                    role="user", content=resource.model_dump_json()
                )
                history.append(Message(role="assistant", tool_calls=resp.tool_calls))
                history.append(Message(role="tool", tool_results=results))
                continue

            # finish_reason == "stop": validate and return
            valid: list[Recommendation] = []
            unsupported: list[tuple[Recommendation, list[str]]] = []
            for rec in resp.recommendations:
                ok, missing = validate_recommendation(rec)
                if ok:
                    valid.append(rec.model_copy(update={"trace_id": trace_id}))
                else:
                    unsupported.append((rec, missing))

            if unsupported and not retried:
                retried = True
                feedback = _format_critique(unsupported)
                history.append(Message(role="user", content=feedback))
                continue

            # After retry, any rec still in `unsupported` is dropped silently.
            return valid

    def _execute_tools(
        self,
        calls: list[ToolCall],
        resource: ResourceSummary,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for tc in calls:
            fn = TOOL_REGISTRY.get(tc.name)
            if fn is None:
                results.append(ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    output=f"unknown tool: {tc.name}", is_error=True,
                ))
                continue
            try:
                args = dict(tc.arguments)
                if tc.name == "check_idle_signals":
                    args = {"resource": resource}
                output = fn(**args)
                payload = (
                    output.model_dump() if hasattr(output, "model_dump")
                    else json.loads(json.dumps(output, default=str))
                )
                results.append(ToolResult(
                    tool_call_id=tc.id, name=tc.name, output=payload,
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    output=f"{type(e).__name__}: {e}", is_error=True,
                ))
        return results


def _augment_resource(
    resource: ResourceSummary,
    calls: list[ToolCall],
    results: list[ToolResult],
) -> ResourceSummary:
    """Merge utilization tool output into the ResourceSummary so later turns see it."""
    by_id = {r.tool_call_id: r for r in results}
    for tc in calls:
        if tc.name != "get_utilization_stats":
            continue
        result = by_id.get(tc.id)
        if result is None or result.is_error:
            continue
        if not isinstance(result.output, dict):
            continue
        try:
            stats = UtilizationStats.model_validate(result.output)
        except Exception:
            continue
        if resource.utilization is None:
            resource = resource.model_copy(update={"utilization": stats})
    return resource


def _format_critique(unsupported: list[tuple[Recommendation, list[str]]]) -> str:
    parts = ["The following recommendations had unsupported numeric claims:"]
    for rec, missing in unsupported:
        parts.append(
            f"- recommendation_id={rec.recommendation_id}: missing evidence for {missing}"
        )
    parts.append("Please re-emit with proper evidence or drop the recommendation.")
    return "\n".join(parts)
