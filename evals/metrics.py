"""Eval metrics: precision, recall, savings accuracy."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseScore:
    case_id: str
    expected_count: int
    actual_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    negative_violations: int


def score_case(
    case_id: str,
    expected: list[dict],
    actual: list[dict],
    negative_assertions: list[str],
) -> CaseScore:
    """expected: list of {type, recommended_instance_type?, min_monthly_savings_usd?, ...}
    actual:   list of Recommendation.model_dump() outputs.
    """
    matched_actual: set[int] = set()
    tp = 0
    for e in expected:
        for i, a in enumerate(actual):
            if i in matched_actual:
                continue
            if _matches(e, a):
                tp += 1
                matched_actual.add(i)
                break
    fp = len(actual) - tp
    fn = len(expected) - tp
    neg_viol = sum(1 for a in actual if a.get("type") in negative_assertions)
    return CaseScore(case_id=case_id, expected_count=len(expected),
                     actual_count=len(actual), true_positives=tp,
                     false_positives=fp, false_negatives=fn,
                     negative_violations=neg_viol)


def _matches(expected: dict, actual: dict) -> bool:
    if expected["type"] != actual.get("type"):
        return False
    if (it := expected.get("recommended_instance_type")) is not None:
        if actual.get("recommended_state", {}).get("instance_type") != it:
            return False
    if (m := expected.get("min_monthly_savings_usd")) is not None:
        if actual.get("monthly_savings_usd", 0) < m:
            return False
    if (mc := expected.get("min_confidence")) is not None:
        if actual.get("confidence", 0) < mc:
            return False
    if (mr := expected.get("max_risk_level")) is not None:
        order = {"low": 0, "medium": 1, "high": 2}
        if order[actual.get("risk_level", "high")] > order[mr]:
            return False
    return True


def aggregate(scores: list[CaseScore]) -> dict:
    tp = sum(s.true_positives for s in scores)
    fp = sum(s.false_positives for s in scores)
    fn = sum(s.false_negatives for s in scores)
    neg_viols = sum(s.negative_violations for s in scores)
    neg_total = sum(1 for s in scores if s.negative_violations >= 0)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "negative_pass_rate": 1.0 - (neg_viols / max(neg_total, 1)),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }
