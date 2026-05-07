"""Langfuse tracer test (opt-in: requires `make langfuse` running locally)."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live


def test_langfuse_tracer_imports_cleanly():
    """Just verify the module imports — does not require Langfuse running."""
    from cost_optimizer.observability.langfuse_tracer import LangfuseTracer  # noqa: F401


@pytest.mark.skipif(
    not os.environ.get("LANGFUSE_PUBLIC_KEY"),
    reason="LANGFUSE_PUBLIC_KEY not set; skip live integration",
)
def test_langfuse_tracer_records_trace():
    from cost_optimizer.observability.langfuse_tracer import LangfuseTracer
    tracer = LangfuseTracer()
    h = tracer.start_trace(resource_id="i-test")
    tracer.record_llm_call(h, prompt="hi", response="ok", tokens=5, latency_ms=100)
    tracer.end_trace(h, recommendations=[], cost_usd=0.0001)
