"""Validates that numeric claims in Recommendation.reasoning are backed by Evidence."""
from __future__ import annotations

import re
from typing import Any

from cost_optimizer.models import Evidence, Recommendation

PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DOLLAR_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
INSTANCE_TYPE_RE = re.compile(r"\b([a-z]\d+\.[a-z0-9]+)\b")

TOLERANCE = 0.05  # 5%


def validate_recommendation(rec: Recommendation) -> tuple[bool, list[str]]:
    """Return (ok, list_of_unsupported_claims).

    A numeric claim is supported when *some* Evidence.data value (recursively)
    is within +/-5% (for numbers) or string-equal (for instance types) to the claim.
    """
    missing: list[str] = []
    text = rec.reasoning

    for m in PERCENT_RE.finditer(text):
        val = float(m.group(1))
        if not _any_number_within(val, rec.evidence, tol=TOLERANCE):
            missing.append(f"{val}% (no matching Evidence value)")

    for m in DOLLAR_RE.finditer(text):
        val = float(m.group(1))
        if not _any_number_within(val, rec.evidence, tol=TOLERANCE):
            missing.append(f"${val} (no matching Evidence value)")

    for m in INSTANCE_TYPE_RE.finditer(text):
        itype = m.group(1)
        if not _any_string_equal(itype, rec.evidence):
            missing.append(f"{itype} (no matching Evidence string)")

    return (not missing, missing)


def _any_number_within(target: float, evidence: list[Evidence], *, tol: float) -> bool:
    for e in evidence:
        if _walk_numbers(e.data, target, tol):
            return True
    return False


def _any_string_equal(target: str, evidence: list[Evidence]) -> bool:
    for e in evidence:
        if _walk_strings(e.data, target):
            return True
    return False


def _walk_numbers(node: Any, target: float, tol: float) -> bool:
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        if target == 0:
            return abs(node) < 1e-6
        return abs(node - target) <= tol * abs(target)
    if isinstance(node, dict):
        return any(_walk_numbers(v, target, tol) for v in node.values())
    if isinstance(node, list):
        return any(_walk_numbers(v, target, tol) for v in node)
    return False


def _walk_strings(node: Any, target: str) -> bool:
    if isinstance(node, str):
        return target in node
    if isinstance(node, dict):
        return any(_walk_strings(v, target) for v in node.values())
    if isinstance(node, list):
        return any(_walk_strings(v, target) for v in node)
    return False
