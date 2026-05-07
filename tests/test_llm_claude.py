"""Claude LLM adapter tests."""
from __future__ import annotations

import os

import pytest


def test_claude_module_imports_cleanly():
    """Module must import even when ANTHROPIC_API_KEY is unset."""
    from cost_optimizer.llm.claude import ClaudeLLM  # noqa: F401


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_claude_complete_against_real_api():
    from cost_optimizer.llm.claude import ClaudeLLM
    from cost_optimizer.models import Message

    llm = ClaudeLLM()
    resp = llm.complete(
        [Message(role="system", content="Reply with the single word PONG."),
         Message(role="user", content="ping")],
        tools=[],
    )
    assert resp.finish_reason in {"tool_use", "stop"}
