"""Batch runner: ingest -> aggregate -> per-resource agent -> aggregate output."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cost_optimizer.agent import Agent
from cost_optimizer.ingest.aggregate import top_n_by_cost
from cost_optimizer.llm.base import LLM
from cost_optimizer.models import Recommendation
from cost_optimizer.observability.base import Tracer
from cost_optimizer.providers.base import BillingProvider


@dataclass
class RunResult:
    recommendations: list[Recommendation] = field(default_factory=list)
    analyzed_count: int = 0
    failed_count: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


class Runner:
    def __init__(
        self,
        provider: BillingProvider,
        llm: LLM,
        *,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider
        self.llm = llm
        self.tracer = tracer

    def run(self, csv_path: Path, *, top_n: int = 50) -> RunResult:
        items = self.provider.parse_csv(Path(csv_path))
        summaries = self.provider.aggregate(items)
        candidates = top_n_by_cost(summaries, n=top_n)

        agent = Agent(llm=self.llm, tracer=self.tracer)
        result = RunResult()
        for r in candidates:
            try:
                recs = agent.run(r)
                result.recommendations.extend(recs)
                result.analyzed_count += 1
            except Exception as e:
                result.failed_count += 1
                result.failures.append((r.resource_id, f"{type(e).__name__}: {e}"))
        return result
