# Cloud Cost Optimizer Agent

A tool-augmented agent that ingests AWS Cost & Usage Report CSVs and produces ranked, evidence-backed cost optimization recommendations.

**Project 1 of the Agentic Patterns Series.** Single-agent ReAct (tool-using loop) with strict structured output, evaluation as a CI gate, and end-to-end observability.

## Why this exists

Most "cost optimizer" demos hallucinate prices and skip the production discipline. This one:

1. **Never invents prices** — the agent calls a pricing tool; numeric claims in the recommendation must be corroborated by the tool's evidence, or the recommendation is dropped.
2. **Strict structured output** — every recommendation is a Pydantic-validated `Recommendation` object.
3. **Evaluation in CI** — a golden set runs on every push; precision/recall thresholds gate the build.
4. **Observability** — every agent run emits a trace (Langfuse for dev, JSONL for CI).

## Quickstart

```bash
# install
make install

# run on the sample CSV (uses MockLLM by default — no API key required)
make run

# run full eval suite (5 cases)
make eval

# bring up Langfuse + Streamlit demo
make langfuse
make demo  # http://localhost:8501

# tests
make test
```

## Architecture

```
CSV → AwsCurParser → BillingLineItems → Aggregator → ResourceSummaries
                                                          │
                                                  per-resource loop
                                                          │
                                            Agent (LLM, tools, tracer)
                                          ┌───────────┴───────────┐
                                       MockLLM              ClaudeLLM
                                                          (Sonnet 4.6)
                                                          │
                                            EvidenceValidator (1 retry)
                                                          │
                                                  list[Recommendation]
```

See [DESIGN.md](DESIGN.md) for the full design document and [docs/superpowers/specs/2026-05-06-cost-optimizer-v1-design.md](docs/superpowers/specs/2026-05-06-cost-optimizer-v1-design.md) for the v1 scope.

## Sample output

```json
{
  "type": "rightsize",
  "resource_id": "i-under-001",
  "recommended_state": {"instance_type": "t3.medium"},
  "monthly_savings_usd": 90.88,
  "annual_savings_usd": 1090.56,
  "confidence": 0.86,
  "risk_level": "medium",
  "reasoning": "Instance shows CPU p95 of 14.0% over 30 days. A t3.medium provides sufficient headroom; current rate is $0.1664/hour and target rate is $0.0416/hour.",
  "evidence": [
    {"description": "30-day CPU p95", "source": "utilization", "data": {"value": 14.0, "unit": "percent"}},
    {"description": "Current on-demand price for t3.xlarge", "source": "pricing_api", "data": {"usd_per_hour": 0.1664}},
    {"description": "Target on-demand price for t3.medium", "source": "pricing_api", "data": {"usd_per_hour": 0.0416}}
  ]
}
```

## Eval results (v1)

Five-case golden set against MockLLM (deterministic):

| Metric | Target | Latest |
|---|---|---|
| Precision | ≥ 0.85 | 1.000 |
| Recall | ≥ 0.80 | 1.000 |
| Negative pass rate | ≥ 0.95 | 1.000 |

Run `make eval` to regenerate; reports land in `evals/reports/`.

## Roadmap

- **v2:** Concrete OCI provider, full 4-tab Streamlit, 50-case golden set, eval-on-PR comment, Loom demo
- **v3:** Real CloudWatch integration, FastAPI surface, Slack alerts, recommendation diffing

## License

MIT
