"""LLM Protocol."""
from __future__ import annotations

from typing import Any, Protocol

from cost_optimizer.models import LLMResponse, Message


class LLM(Protocol):
    """An LLM that can complete a message history with tool-call awareness."""

    name: str

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...
