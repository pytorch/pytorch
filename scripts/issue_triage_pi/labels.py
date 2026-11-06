"""Label rules for the issue-triage apply step."""

from __future__ import annotations

import json
import re
from pathlib import Path


SKILLS = Path(__file__).resolve().parents[2] / ".agents/skills"
TRIAGE_SKILL = SKILLS / "triaging-issues"
DISTRIBUTED = "oncall: distributed"
PARALLELISMS = "oncall: distributed parallelisms"
SUB_QUEUES = {
    "oncall: distributed checkpointing",
    "oncall: distributed infra",
    PARALLELISMS,
}
# Distributed component labels; parallelisms needs at least one to complete triage.
DISTRIBUTED_MODULES = {
    "module: activation checkpointing",
    "module: c10d",
    "module: context parallel",
    "module: data parallel",
    "module: ddp",
    "module: DeviceMesh",
    "module: distributed_tool",
    "module: dtensor",
    "module: elastic",
    "module: fsdp",
    "module: mpi",
    "module: nccl",
    "module: pipelining",
    "module: rpc",
    "module: symm_mem",
    "module: threaded pg",
}

# Labels the triage bot must never add: CI controls, release notes, severity,
# and decisions reserved for human reviewers.
FORBIDDEN_PATTERNS = [
    r"^ciflow/",
    r"^test-config/",
    r"^release notes:",
    r"^ci-",
    r"^ci:",
    r"^sev",
    r"deprecated",
]
FORBIDDEN_EXACT = {
    "actionable",
    "high priority",
    "merge blocking",
    "needs design",
    "needs reproduction",
    "needs research",
    "oncall: releng",  # Not a triage redirect target; use module: ci instead
}
# (specific, general): drop the general label when both are requested.
REDUNDANT_PAIRS = [
    ("module: rnn", "module: nn"),
]


def is_forbidden(label: str) -> bool:
    lowered = label.lower()
    return lowered in FORBIDDEN_EXACT or any(
        re.search(pattern, lowered) for pattern in FORBIDDEN_PATTERNS
    )


def load_valid_labels() -> set[str]:
    """Labels the issue-triage skill may request."""
    catalog = json.loads((TRIAGE_SKILL / "labels.json").read_text())
    return {label["name"] for label in catalog["labels"]}


def load_templates(skill: Path) -> dict[str, dict]:
    return json.loads((skill / "templates.json").read_text())["templates"]


def strip_redundant(labels: list[str]) -> tuple[list[str], list[str]]:
    present = set(labels)
    redundant = {
        general
        for specific, general in REDUNDANT_PAIRS
        if {specific, general} <= present
    }
    return [label for label in labels if label not in redundant], sorted(redundant)
