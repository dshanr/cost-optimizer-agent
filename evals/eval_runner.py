"""Run the agent against the golden set, score, emit markdown report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cost_optimizer.agent import Agent
from cost_optimizer.llm.base import LLM
from cost_optimizer.models import ResourceSummary
from evals.metrics import CaseScore, aggregate, score_case


@dataclass
class EvalReport:
    case_count: int
    precision: float
    recall: float
    negative_pass_rate: float
    case_scores: list[CaseScore]
    markdown_path: Path | None = None


def run_eval(
    *,
    llm: LLM,
    golden_set_path: Path | None = None,
    output_dir: Path | None = None,
) -> EvalReport:
    golden_set_path = golden_set_path or (
        Path(__file__).resolve().parent / "golden_set.json"
    )
    output_dir = output_dir or (Path(__file__).resolve().parent / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads(golden_set_path.read_text())["cases"]
    agent = Agent(llm=llm)
    case_scores: list[CaseScore] = []

    for case in cases:
        resource = ResourceSummary.model_validate(case["input_resource"])
        recs = agent.run(resource)
        score = score_case(
            case_id=case["case_id"],
            expected=case["expected_recommendations"],
            actual=[r.model_dump() for r in recs],
            negative_assertions=case.get("negative_assertions", []),
        )
        case_scores.append(score)

    agg = aggregate(case_scores)
    report = EvalReport(
        case_count=len(cases),
        precision=agg["precision"],
        recall=agg["recall"],
        negative_pass_rate=agg["negative_pass_rate"],
        case_scores=case_scores,
    )
    report.markdown_path = _write_markdown(report, output_dir)
    return report


def _write_markdown(report: EvalReport, output_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = output_dir / f"eval-{ts}.md"
    body = (
        f"# Eval Report — {ts}\n\n"
        f"- **Cases:** {report.case_count}\n"
        f"- **Precision:** {report.precision:.3f}\n"
        f"- **Recall:** {report.recall:.3f}\n"
        f"- **Negative pass rate:** {report.negative_pass_rate:.3f}\n\n"
        f"| case_id | expected | actual | tp | fp | fn |\n"
        f"|---|---|---|---|---|---|\n"
    )
    for s in report.case_scores:
        body += (
            f"| {s.case_id} | {s.expected_count} | {s.actual_count} "
            f"| {s.true_positives} | {s.false_positives} | {s.false_negatives} |\n"
        )
    path.write_text(body)
    return path


def main() -> int:
    from cost_optimizer.llm.mock import MockLLM
    report = run_eval(llm=MockLLM())
    print(f"Cases: {report.case_count}  precision={report.precision:.3f}  "
          f"recall={report.recall:.3f}")
    print(f"Report: {report.markdown_path}")
    if report.precision < 0.85 or report.recall < 0.80:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
