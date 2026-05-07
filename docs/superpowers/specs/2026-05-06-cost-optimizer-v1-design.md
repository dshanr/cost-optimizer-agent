# Cost Optimizer Agent — v1 Design (lean)

**Date:** 2026-05-06
**Source spec:** [DESIGN.md](../../../DESIGN.md) (full portfolio target)
**This document:** scoped, TDD-first v1 plan derived from DESIGN.md

---

## 1. Scope

A CLI-first cloud cost optimization agent that:

1. Ingests AWS Cost & Usage Report (CUR) CSVs.
2. Aggregates line items into per-resource summaries.
3. Runs a tool-augmented ReAct loop per resource (default `MockLLM`, opt-in `ClaudeLLM`).
4. Validates each `Recommendation` against an evidence-corroboration rule.
5. Emits validated recommendations to stdout (JSONL) and a trace store.
6. Exposes a minimal Streamlit UI for upload → run → drill-down.
7. Captures every run in Langfuse (self-hosted via Docker Compose).

### In scope for v1

- AWS provider (concrete implementation).
- OCI provider (Protocol + contract tests; `NotImplementedError` stub).
- `MockLLM` (default, deterministic, used in all automated tests).
- `ClaudeLLM` (Anthropic Sonnet 4.6, opt-in, tested with `pytest.mark.live`).
- Six tools from DESIGN.md §8 (pricing, utilization, rightsizing options, commitment savings, idle signals).
- Pricing tool defaults to disk fixtures; live API only with `COST_OPTIMIZER_LIVE_PRICING=1`.
- Utilization is mocked with deterministic synthetic patterns keyed off `hash(resource_id)`.
- Five-case golden eval set + eval runner + markdown report.
- `Tracer` protocol with `JsonlTracer` (tests/CI) and `LangfuseTracer` (dev/demo).
- `docker-compose.yml` for Langfuse + Postgres.
- Minimal Streamlit app: single recommendations tab with file upload, table, drill-down, trace links.
- GitHub Actions running pytest on push.

### Explicitly out of scope for v1

- Concrete OCI provider implementation (Protocol only).
- 50-case golden set (5 cases in v1; expansion in v2).
- Full 4-tab Streamlit (only recommendations tab in v1).
- `eval.yml` GitHub Action posting markdown to PR comments (v2).
- Real CloudWatch / OCI Monitoring integration.
- Loom demo recording.
- FastAPI surface, Slack alerts, anomaly detection (DESIGN.md §17).

---

## 2. Definition of Done (v1)

- [ ] `make install` succeeds; `make test` passes with ≥85% coverage on `src/`.
- [ ] `make eval` runs the 5-case golden set against `MockLLM`; precision ≥ 0.85, recall ≥ 0.80.
- [ ] `cost-optimizer run data/sample_aws_cur.csv` prints valid `Recommendation` JSONL.
- [ ] With `ANTHROPIC_API_KEY` set, `cost-optimizer run --llm claude data/sample_aws_cur.csv` works.
- [ ] `make langfuse && make demo` brings up Langfuse and Streamlit; uploading the sample CSV produces recommendations with working Langfuse trace links.
- [ ] GitHub Actions `test.yml` is green on a test PR.
- [ ] README v1 has: hook, quickstart, architecture diagram, sample output, link to DESIGN.md and this spec.

---

## 3. Architecture

```
CSV ──► AwsCurParser ──► list[BillingLineItem] ──► Aggregator ──► list[ResourceSummary]
                                                                          │
                                                                          ▼
                                                                  per-resource loop
                                                                          │
                                                                          ▼
                                                              Agent(LLM, tools, tracer)
                                                              │           │
                                                              ▼           ▼
                                                          MockLLM    ClaudeLLM
                                                          (default)  (opt-in)
                                                                          │
                                                                          ▼
                                                          EvidenceValidator (1 retry)
                                                                          │
                                                                          ▼
                                                          list[Recommendation]
                                                                          │
                                            ┌─────────────────────────────┼─────────────────────────────┐
                                            ▼                             ▼                             ▼
                                         stdout JSONL              Tracer (Jsonl|Langfuse)        Streamlit UI
```

### Key invariants

1. The LLM only ever sees a single `ResourceSummary` per agent invocation.
2. The ReAct loop is hard-capped at 6 tool calls per resource.
3. Every numeric claim in `Recommendation.reasoning` must be supported by an `Evidence` entry; unsupported recommendations are rejected (1 retry, then dropped).
4. Per-resource failures are isolated: a crash on resource N+1 does not affect recommendations from resource N.
5. The Pricing API is not hit in tests or CI by default; tests use snapshot fixtures.
6. Real LLM calls (`ClaudeLLM`) and real Pricing API calls are gated by `pytest.mark.live` and skipped in CI.

---

## 4. Components

### 4.1 Models (`src/cost_optimizer/models.py`)

Pydantic v2 models per DESIGN.md §6:

- `BillingLineItem`
- `ResourceSummary`
- `UtilizationStats`
- `Recommendation` (with `RecommendationType` enum, `Evidence`)
- `LLMResponse` (new in v1: structured wrapper for an LLM completion containing tool calls and/or final recommendations)

Validators:
- `confidence` ∈ [0.0, 1.0]
- `monthly_savings_usd × 12 ≈ annual_savings_usd` within $1 tolerance
- `recommendation_id` is a UUID4
- `risk_level`, `effort` constrained to literals

### 4.2 Ingest (`src/cost_optimizer/ingest/`)

- **`aws_cur.py`** — Parses AWS CUR CSV (subset of fields: `lineItem/UsageStartDate`, `lineItem/UsageEndDate`, `lineItem/ResourceId`, `lineItem/UnblendedCost`, `lineItem/UsageAmount`, `lineItem/UsageType`, `product/instanceType`, `product/region`, `lineItem/UsageAccountId`, plus `resourceTags/*`). Emits `list[BillingLineItem]`. Raises `IngestError` with line number on malformed rows.
- **`aggregate.py`** — Groups by `(provider, resource_id)`, sums `unblended_cost_usd`, computes `usage_hours`, attaches utilization (lazily, via tool call). Returns `list[ResourceSummary]` sorted by `monthly_cost_usd` desc. Top-N filter applied here.

### 4.3 Tools (`src/cost_optimizer/tools/`)

Per DESIGN.md §8:

- **`pricing.py`** — `get_aws_pricing(instance_type, region, os, tenancy) -> AwsPricingResult`. Fixture-backed by default (`tests/fixtures/pricing/aws/<region>/<instance_type>.json`); live mode under `COST_OPTIMIZER_LIVE_PRICING=1`. `get_oci_pricing` exists as a stub raising `NotImplementedError`.
- **`utilization.py`** — `get_utilization_stats(resource_id, provider, days)`. Deterministic synthetic data keyed off `hash(resource_id)`. Distribution: 30% clearly idle, 30% clearly hot, 40% normal-utilization. Patterns are stable across runs.
- **`rightsizing.py`** — `get_rightsizing_options(instance_type, target_cpu_utilization)`. Reads from `data/instance_catalog.json` (static).
- **`savings.py`** — `calculate_commitment_savings(...)`. Pure calculation; calls `get_aws_pricing` internally.
- **`idle.py`** — `check_idle_signals(resource)`. Heuristic detector: stopped EC2 with EBS attached, unattached EBS, idle ELB, unused EIP.

Each tool returns a typed Pydantic result.

### 4.4 Providers (`src/cost_optimizer/providers/`)

- **`base.py`** — `BillingProvider` Protocol with `parse_csv`, `aggregate`, `get_pricing_tool`, `get_utilization_tool`, `name`.
- **`aws.py`** — concrete impl wired to `ingest/aws_cur.py` and `tools/pricing.get_aws_pricing`.
- **`oci.py`** — stub class; methods raise `NotImplementedError`.

`tests/test_providers.py` defines a contract test parametrized over all known providers; AWS passes, OCI is `pytest.mark.xfail(strict=True)`.

### 4.5 LLM (`src/cost_optimizer/llm/`)

- **`base.py`** — `LLM` Protocol:
  ```python
  class LLM(Protocol):
      name: str
      def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse: ...
  ```
- **`mock.py`** — `MockLLM`. Deterministic responses keyed off `(resource_type, monthly_cost_usd, utilization_p95_cpu)`. Hand-coded scenarios cover all five recommendation types. Used in every automated test.
- **`claude.py`** — `ClaudeLLM`. Uses `anthropic` SDK; model `claude-sonnet-4-6`. Translates between our `Message`/`ToolSpec` types and Anthropic's. Opt-in only.

### 4.6 Agent (`src/cost_optimizer/agent.py`)

LangGraph single-node ReAct loop:

1. Receive `ResourceSummary`.
2. Render system prompt (`prompts/system.md`) with tool list + schema + the resource as input.
3. Call `LLM.complete`. If response contains tool calls → execute → append result to messages → loop. Cap: 6 iterations.
4. When LLM returns recommendations (or after iteration cap), pass them through `EvidenceValidator`.
5. On validation failure: send a structured critique back to the LLM, allow one retry. On second failure: drop the offending recommendation, log to trace.
6. Return `list[Recommendation]`.

### 4.7 Evidence validator (`src/cost_optimizer/evidence_validator.py`)

- Regex-extract numeric claims from `Recommendation.reasoning`: percentages (`\d+(?:\.\d+)?\s*%`), dollar amounts (`\$\d+(?:\.\d+)?`), instance types (`[a-z]\d+\.[a-z]+`).
- For each claim, check if any `Evidence.data` value (recursively) contains a matching number (within 5% tolerance for percentages and dollars) or string (for instance types).
- Return `(ok: bool, unsupported_claims: list[str])`.
- Validator is pure; agent owns the retry logic.

### 4.8 Runner & CLI (`src/cost_optimizer/runner.py`, `src/cost_optimizer/cli.py`)

- **`runner.py`** — `Runner.run(csv_path, *, top_n, llm, tracer) -> RunResult`. Orchestrates ingest → aggregate → top-N → per-resource agent → emit.
- **`cli.py`** — `typer` app:
  - `cost-optimizer run <csv> [--top-n 50] [--llm mock|claude] [--output jsonl|json]`
  - `cost-optimizer eval [--baseline evals/baseline.json]`

### 4.9 Observability (`src/cost_optimizer/observability/`)

- **`base.py`** — `Tracer` Protocol:
  ```python
  class Tracer(Protocol):
      def start_trace(self, *, resource_id: str) -> TraceHandle: ...
      def record_llm_call(self, handle: TraceHandle, *, prompt, response, tokens, latency_ms): ...
      def record_tool_call(self, handle: TraceHandle, *, tool, input, output, latency_ms): ...
      def end_trace(self, handle: TraceHandle, *, recommendations, cost_usd): ...
  ```
- **`jsonl_tracer.py`** — Writes one JSON line per trace to `runs/<timestamp>.jsonl`. Used in tests and CI.
- **`langfuse_tracer.py`** — Adapter to Langfuse SDK. Used in dev/demo.

### 4.10 Streamlit (`app/streamlit_app.py`)

Single tab in v1: **Recommendations**.
- Sidebar: file upload, top-N slider, LLM selector (`mock`/`claude`), Run button.
- Main: table of recommendations sorted by `annual_savings_usd` desc.
- Click a row → expander shows `reasoning`, `evidence`, `prerequisites`, `rollback_plan`, and a clickable Langfuse trace URL.

The full 4-tab UI from DESIGN.md §12 lands in v2.

### 4.11 Eval (`evals/`)

- **`golden_set.json`** — 5 cases:
  1. `aws-rightsize-001` — t3.xlarge with 14% p95 CPU → recommend t3.medium.
  2. `aws-idle-001` — stopped EC2 with attached EBS → recommend `terminate_idle`.
  3. `aws-commitment-001` — steady m5.large in prod, 24/7 → recommend 1y RI.
  4. `aws-storage-001` — S3 Standard with low access pattern → recommend Glacier transition.
  5. `aws-negative-001` — t3.medium with 78% p95 CPU → no recommendation (already right-sized).
- **`eval_runner.py`** — Loads golden set, runs `Agent(MockLLM)` on each, compares to expected, computes precision/recall/savings-accuracy/calibration. Emits `evals/reports/<timestamp>.md`.
- **`metrics.py`** — Pure functions, unit-tested.
- **`baseline.json`** — Stored last-accepted scores; comparison enabled via `--baseline`.

---

## 5. Build Sequence (TDD)

Strict TDD: failing test → implementation → green → commit. Each step is independently committable. Estimated total: **~15h**.

| # | Increment | Time | Acceptance |
|---|---|---|---|
| 0 | Scaffold: `uv init`, `pyproject.toml`, Makefile, `.env.example`, commit DESIGN.md + this spec | 0.5h | `make install` works, `pytest` runs (zero tests) |
| 1 | `models.py` + tests (round-trip, validators, edge cases) | 1h | `pytest tests/test_models.py` green |
| 2 | `ingest/aws_cur.py` + 12-row fixture CSV + tests | 1h | parses fixture, raises on malformed |
| 3 | `ingest/aggregate.py` + tests | 0.5h | grouping, top-N, edge cases |
| 4 | All six tools + fixtures + tests | 2h | each tool has ≥2 tests |
| 5 | `providers/base.py` + `providers/aws.py` + `providers/oci.py` (stub) + contract tests | 0.5h | AWS passes; OCI `xfail` |
| 6 | `llm/base.py` + `llm/mock.py` + tests | 1h | MockLLM returns expected tool-call sequences |
| 7 | `evidence_validator.py` + tests | 1h | rejects unsupported claims, accepts well-formed |
| 8 | `agent.py` + integration tests with MockLLM on all 5 golden cases | 1.5h | all 5 cases produce expected recommendations |
| 9 | `runner.py` + `cli.py` + CliRunner tests | 1h | `cost-optimizer run sample.csv` works end-to-end |
| 10 | `observability/base.py` + `jsonl_tracer.py` + wired into agent + tests | 0.5h | trace file written per run |
| 11 | `langfuse_tracer.py` + `docker-compose.yml` + `make langfuse` + opt-in test | 1h | Langfuse trace appears when running locally |
| 12 | `evals/` golden set + eval_runner + metrics + report generator | 1.5h | `make eval` produces markdown report meeting thresholds |
| 13 | `llm/claude.py` adapter + opt-in `pytest.mark.live` test | 0.5h | imports cleanly; live test works with API key |
| 14 | `app/streamlit_app.py` minimal (recommendations tab only) | 1.5h | upload → run → drill-down → trace link works |
| 15 | GitHub Actions `test.yml` (pytest on push) + README v1 | 1h | green CI on a test PR; README lets stranger run the demo |

---

## 6. Stack

- Python 3.11+ (target 3.11 for portability; 3.14 available locally)
- `uv` for dep + venv management
- `pydantic>=2.5`
- `langgraph` (single-node now; portfolio coherence with Tier 2–4 projects)
- `anthropic` (Claude adapter)
- `typer` (CLI)
- `httpx` + `diskcache` (pricing API)
- `streamlit`
- `langfuse` (Python SDK)
- `pytest`, `pytest-cov`, `pytest-mock`, `pytest-asyncio`
- Dev: `ruff`, `mypy`

---

## 7. Repo Layout

```
cost-optimizer-agent/
├── README.md
├── DESIGN.md
├── docs/superpowers/specs/2026-05-06-cost-optimizer-v1-design.md
├── pyproject.toml
├── Makefile
├── .env.example
├── docker-compose.yml
├── .github/workflows/test.yml
│
├── src/cost_optimizer/
│   ├── __init__.py
│   ├── models.py
│   ├── agent.py
│   ├── evidence_validator.py
│   ├── runner.py
│   ├── cli.py
│   ├── prompts/system.md
│   ├── ingest/
│   │   ├── aws_cur.py
│   │   └── aggregate.py
│   ├── tools/
│   │   ├── pricing.py
│   │   ├── utilization.py
│   │   ├── rightsizing.py
│   │   ├── savings.py
│   │   └── idle.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── aws.py
│   │   └── oci.py
│   ├── llm/
│   │   ├── base.py
│   │   ├── mock.py
│   │   └── claude.py
│   └── observability/
│       ├── base.py
│       ├── jsonl_tracer.py
│       └── langfuse_tracer.py
│
├── app/streamlit_app.py
│
├── evals/
│   ├── golden_set.json
│   ├── eval_runner.py
│   ├── metrics.py
│   ├── baseline.json
│   └── reports/
│
├── tests/
│   ├── fixtures/
│   │   ├── pricing/aws/...
│   │   └── csv/sample_aws_cur.csv
│   ├── test_models.py
│   ├── test_ingest.py
│   ├── test_tools.py
│   ├── test_providers.py
│   ├── test_llm_mock.py
│   ├── test_evidence_validator.py
│   ├── test_agent.py
│   ├── test_runner.py
│   ├── test_cli.py
│   ├── test_observability.py
│   └── test_eval_runner.py
│
└── data/
    ├── sample_aws_cur.csv
    └── instance_catalog.json
```

---

## 8. Trade-offs and Decisions

| Decision | Choice | Why |
|---|---|---|
| TDD strictness | Strict | User explicitly required test-driven for every step. |
| LLM default | `MockLLM` | TDD only works with deterministic LLM; real LLM is opt-in. |
| Pricing source | Disk fixtures by default | CI must not depend on network. |
| OCI provider | Protocol + contract tests, no concrete impl | Multi-cloud abstraction is exercised; concrete v2. |
| Langfuse | Self-hosted from v1 | User opted in (option C). Trace links land in Streamlit from day one. |
| Streamlit | Single tab in v1 | User opted in (option C). Full 4-tab in v2. |
| Golden set size | 5 cases | Enough to prove harness; expansion is mechanical and v2. |
| LangGraph vs Pydantic AI | LangGraph | DESIGN.md §5 — portfolio coherence with later projects. |
| Per-resource vs batch agent | Per-resource | Bounded cost, parallelizable, isolated failures. |
| Read-only by design | Yes | DESIGN.md §16. v1 advises, never executes. |

---

## 9. v2 Roadmap (post-v1)

| # | Add | Time |
|---|---|---|
| 1 | Expand golden set 5 → 50 cases | 2h |
| 2 | `eval.yml` GitHub Action posting markdown report as PR comment | 0.5h |
| 3 | Full Streamlit 4-tab UI (resource browser, run summary, about) | 2h |
| 4 | OCI concrete provider (`providers/oci.py`, `ingest/oci_billing.py`, OCI pricing tool) | 2h |
| 5 | Loom recording + README polish | 1h |

---

## 10. v3 Stretch (DESIGN.md §17)

- Real CloudWatch / OCI Monitoring integration
- FastAPI REST surface
- Slack notification integration
- Recommendation diffing across runs
- Multi-account aggregation

Each is independently optional and ~2–4h.
