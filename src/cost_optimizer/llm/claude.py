"""Anthropic Claude adapter implementing the LLM Protocol.

Uses the Messages API with tool use. Translates between our Message/ToolCall
types and Anthropic's. Caches the system prompt with prompt caching for cost
predictability across the per-resource loop.
"""
from __future__ import annotations

import json
import os
from typing import Any

from cost_optimizer.models import (
    LLMResponse,
    Message,
    Recommendation,
    ToolCall,
)

MODEL_ID = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 4096


class ClaudeLLM:
    name = "claude"

    def __init__(self, *, model: str = MODEL_ID) -> None:
        # Import lazily so the module imports without the SDK installed at all.
        from anthropic import Anthropic
        self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        system_text = ""
        api_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_text = m.content or ""
                continue
            api_msgs.append(_to_api_message(m))

        anth_tools = _claude_tool_specs()

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=[{"type": "text", "text": system_text,
                     "cache_control": {"type": "ephemeral"}}] if system_text else [],
            messages=api_msgs,
            tools=anth_tools,
        )

        tool_calls: list[ToolCall] = []
        text_chunks: list[str] = []
        for block in resp.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input),
                ))
            elif block.type == "text":
                text_chunks.append(block.text)

        text = "\n".join(text_chunks)
        recommendations = _parse_recommendations(text)

        if tool_calls and not recommendations:
            return LLMResponse(tool_calls=tool_calls, finish_reason="tool_use", raw_text=text)
        return LLMResponse(
            recommendations=recommendations,
            finish_reason="stop",
            raw_text=text,
        )


def _to_api_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": json.dumps(tr.output) if not isinstance(tr.output, str) else tr.output,
                    "is_error": tr.is_error,
                }
                for tr in m.tool_results
            ],
        }
    if m.tool_calls:
        return {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content or ""}


def _claude_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_aws_pricing",
            "description": "Fetch on-demand and commitment pricing for an EC2 instance type.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "instance_type": {"type": "string"},
                    "region": {"type": "string"},
                },
                "required": ["instance_type", "region"],
            },
        },
        {
            "name": "get_utilization_stats",
            "description": "Fetch CPU/memory/network utilization percentiles for a resource.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string"},
                    "provider": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["resource_id", "provider"],
            },
        },
        {
            "name": "get_rightsizing_options",
            "description": "List smaller instance types in the same family.",
            "input_schema": {
                "type": "object",
                "properties": {"instance_type": {"type": "string"}},
                "required": ["instance_type"],
            },
        },
        {
            "name": "calculate_commitment_savings",
            "description": "Compute RI/Savings Plan savings for a steady-state workload.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "monthly_on_demand_cost_usd": {"type": "number"},
                    "instance_type": {"type": "string"},
                    "region": {"type": "string"},
                    "term_years": {"type": "integer"},
                },
                "required": ["monthly_on_demand_cost_usd", "instance_type", "region"],
            },
        },
        {
            "name": "check_idle_signals",
            "description": "Heuristic detector for idle/orphaned resources.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ]


def _parse_recommendations(text: str) -> list[Recommendation]:
    """Recommendations may be returned as a JSON array embedded in the text."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    out: list[Recommendation] = []
    for item in payload:
        try:
            out.append(Recommendation.model_validate(item))
        except Exception:
            continue
    return out
