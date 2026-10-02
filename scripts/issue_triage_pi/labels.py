"""Label rules shared by the issue-triage and distributed-triage apply steps."""

from __future__ import annotations

import json
import re
from pathlib import Path


SKILLS = Path(__file__).resolve().parents[2] / ".agents/skills"
TRIAGE_SKILL = SKILLS / "triaging-issues"
DISTRIBUTED_SKILL = SKILLS / "distributed-triage"
DISTRIBUTED = "oncall: distributed"

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


def is_sub_queue(label: str) -> bool:
    """`oncall: distributed <team>`, picked by distributed triage only."""
    return label.startswith(DISTRIBUTED + " ")


def load_labels(path: Path) -> set[str]:
    return {label["name"] for label in json.loads(path.read_text())["labels"]}


def load_valid_labels() -> set[str]:
    """Labels the issue-triage skill may request."""
    return load_labels(TRIAGE_SKILL / "labels.json") | load_labels(
        DISTRIBUTED_SKILL / "distributed-labels.json"
    )


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
