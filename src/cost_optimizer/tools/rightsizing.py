"""Rightsizing options from a static instance catalog."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class InstanceOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_type: str
    vcpu: int
    memory_gib: float


class UnknownInstanceFamilyError(LookupError):
    pass


def get_rightsizing_options(
    instance_type: str,
    target_cpu_utilization: float = 0.6,
) -> list[InstanceOption]:
    """Return smaller instances in the same family as `instance_type`.

    Sorted by vCPU ascending. Excludes the input instance and any that are
    larger. `target_cpu_utilization` is accepted but not yet used to filter
    (the agent reasons about headroom from utilization data).
    """
    family = instance_type.split(".")[0]
    catalog = _catalog()
    if family not in catalog:
        raise UnknownInstanceFamilyError(f"unknown family '{family}' for {instance_type}")

    options = catalog[family]
    current = next((o for o in options if o.instance_type == instance_type), None)
    if current is None:
        raise UnknownInstanceFamilyError(
            f"unknown instance type '{instance_type}' in family '{family}'"
        )

    smaller = [o for o in options if o.vcpu < current.vcpu]
    smaller.sort(key=lambda o: o.vcpu)
    return smaller


@lru_cache(maxsize=1)
def _catalog() -> dict[str, list[InstanceOption]]:
    path = Path(__file__).resolve().parents[3] / "data" / "instance_catalog.json"
    raw = json.loads(path.read_text())
    return {
        family: [InstanceOption.model_validate(o) for o in entries]
        for family, entries in raw.items()
    }
