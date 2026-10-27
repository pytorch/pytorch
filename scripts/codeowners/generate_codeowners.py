#!/usr/bin/env python3
"""Generate CODEOWNERS from OWNERSHIP.yaml.

usage: generate_codeowners.py [--check | --lintrunner]
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


SECTION_MARKER = "# Grouped review-surface definitions:"
LINTER_CODE = "CODEOWNERS_GENERATED"


@dataclass(frozen=True)
class PathRule:
    path: str
    # Handles GitHub asks for review; empty for a commented (quiet) rule.
    notify: tuple[str, ...]
    # CODEOWNERS section the rule is written under (kept for owners.py).
    section: str


@dataclass(frozen=True)
class LabelGroup:
    """Paths owned together, identified by their label set (empty: not owned yet)."""

    labels: tuple[str, ...]
    owners: tuple[str, ...]
    paths: tuple[PathRule, ...]


@dataclass(frozen=True)
class OwnershipFile:
    oncall_queues: dict[str, tuple[str, ...]]
    planned_labels: tuple[str, ...]
    sections: tuple[str, ...]
    section_notes: dict[str, tuple[str, ...]]
    groups: tuple[LabelGroup, ...]


def parse_path_rule(entry, *, section: str) -> PathRule:
    if isinstance(entry, str):
        return PathRule(entry, (), section)
    ((path, value),) = entry.items()
    if isinstance(value, list):
        return PathRule(path, tuple(value), section)
    return PathRule(path, tuple(value.get("notify", ())), value["section"])


def load_ownership(path: Path) -> OwnershipFile:
    document = yaml.safe_load(path.read_text())
    return OwnershipFile(
        oncall_queues={k: tuple(v) for k, v in document["oncall_queues"].items()},
        planned_labels=tuple(document["planned_labels"]),
        sections=tuple(document["sections"]),
        section_notes={k: tuple(v) for k, v in document["section_notes"].items()},
        groups=tuple(
            LabelGroup(
                labels=tuple(entry.get("labels", ())),
                owners=tuple(entry.get("owners", ())),
                paths=tuple(
                    parse_path_rule(rule, section=entry["section"])
                    for rule in entry["paths"]
                ),
            )
            for entry in document["ownership"]
        ),
    )


def handle(name: str) -> str:
    """Handles are stored without "@" (as in merge_rules.yaml); emails keep theirs."""
    return name if "@" in name else f"@{name}"


def render_rule(rule: PathRule) -> str:
    if not rule.notify:
        return f"# {rule.path}"
    return " ".join([rule.path, *map(handle, rule.notify)])


def render_header(ownership: OwnershipFile) -> list[str]:
    oncall_map = [
        f"#     {oncall}: {', '.join(m.removeprefix('module: ') for m in modules)}"
        for oncall, modules in ownership.oncall_queues.items()
    ]
    return [
        "# This file is generated from OWNERSHIP.yaml by",
        "# scripts/codeowners/generate_codeowners.py. Edit OWNERSHIP.yaml and rerun",
        "# the script (or `lintrunner -a`) instead of editing this file.",
        "",
        "# IMPORTANT:",
        "# This file is ONLY used to subscribe for notifications for PRs",
        "# related to a specific file path. Approvals from people in this",
        "# file are not required for merges.",
        "",
        "# CODEOWNERS semantics:",
        "# - GitHub uses the last matching pattern for each path.",
        "# - Active rules name who gets review requests. A commented rule adds no",
        "#   notification of its own; the path follows the last matching active rule,",
        "#   if there is one. Commenting a rule says nothing about ownership.",
        '# - An "# owner:" line records the group meant to own the paths below it, up',
        '#   to the next "# owner:" or "# label:" line or the end of the section. These',
        "#   lines are informational: nothing consumes them yet, and they don't change",
        '#   who is notified. "none" means no owning group is recorded yet.',
        "# - Sections are ordered from broad defaults to narrower overrides.",
        '# - A "# label:" line names the GitHub issue/PR labels for the paths below',
        '#   it, up to the next "# label:" line or the end of the section. Paths',
        '#   before the first "# label:" line in a section have no single label.',
        "# - Oncall queues are implied by module labels and are not repeated on label",
        "#   lines; a label line keeps an oncall only when none of its modules implies",
        "#   it:",
        *oncall_map,
        "# - Labels planned but not yet created on GitHub (issues can't be routed to",
        f"#   them until they exist): {', '.join(ownership.planned_labels)}",
        "#",
        "# For module labels => owners mapping, please see https://github.com/pytorch/pytorch/issues/24422.",
        "",
        "# CODEOWNERS review-surface taxonomy",
        "#",
        "# Every tracked path resolves to one review surface. Every listed pattern must",
        "# match at least one committed path. The fast lintrunner set enforces these",
        "# rules, declared sections, valid owners, source order, no duplicate or",
        '# ineffective patterns, and the "# owner:" rules: every "# label:" line is',
        '# directly followed by one "# owner:" line, a label list has one owner list,',
        "# and an active rule's handles appear on its block's owner line.",
        "#",
        "# Sections are kept temporarily so scripts/codeowners/owners.py keeps its",
        "# section-name fallback; ownership is recorded per label block, not per section.",
        "# Review-surface hierarchy:",
        f"# - Sections: {', '.join(ownership.sections)}",
        "#",
        SECTION_MARKER,
        "#",
    ]


def contains(broad: str, narrow: str) -> bool:
    """True if every path matching `narrow` also matches the directory pattern `broad`."""
    return (
        broad.endswith("/")
        and "*" not in broad
        and narrow != broad
        and narrow.startswith(broad)
    )


Block = tuple[LabelGroup, list[PathRule]]


def order_blocks(blocks: list[Block]) -> list[Block]:
    """Order a section's label blocks so broader patterns come before narrower ones.

    GitHub applies the last matching rule, so a block containing a directory that
    another block's patterns sit inside must come first. Otherwise blocks keep
    their OWNERSHIP.yaml order.
    """

    def precedes(first: Block, second: Block) -> bool:
        return any(contains(a.path, b.path) for a in first[1] for b in second[1])

    remaining = list(blocks)
    ordered = []
    while remaining:
        ready = next(
            (
                block
                for block in remaining
                if not any(precedes(other, block) for other in remaining if other is not block)
            ),
            None,
        )
        if ready is None:
            groups = "; ".join(", ".join(group.labels) for group, _ in remaining)
            raise ValueError(f"label groups nest inside each other both ways: {groups}")
        ordered.append(ready)
        remaining.remove(ready)
    return ordered


def render_codeowners(ownership: OwnershipFile) -> str:
    lines = render_header(ownership)
    for section in ownership.sections:
        lines.append(f"# [{section}]")
        lines.extend(ownership.section_notes.get(section, ()))
        for group in ownership.groups:
            if not group.labels:
                lines.extend(
                    render_rule(rule) for rule in group.paths if rule.section == section
                )
        blocks = [
            (group, [rule for rule in group.paths if rule.section == section])
            for group in ownership.groups
            if group.labels
        ]
        for group, rules in order_blocks([block for block in blocks if block[1]]):
            lines.append(f"# label: {', '.join(group.labels)}")
            lines.append(f"# owner: {' '.join(map(handle, group.owners)) or 'none'}")
            lines.extend(map(render_rule, rules))
        lines.append("#")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if CODEOWNERS is stale")
    mode.add_argument("--lintrunner", action="store_true")
    args = parser.parse_args()
    repo = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    codeowners = repo / "CODEOWNERS"
    generated = render_codeowners(load_ownership(repo / "OWNERSHIP.yaml"))
    current = codeowners.read_text()
    if args.lintrunner:
        if current != generated:
            lint_message = {
                "path": "CODEOWNERS",
                "line": None,
                "char": None,
                "code": LINTER_CODE,
                "severity": "error",
                "name": "stale-codeowners",
                "original": current,
                "replacement": generated,
                "description": (
                    "CODEOWNERS is generated from OWNERSHIP.yaml. Run "
                    "`python scripts/codeowners/generate_codeowners.py` or "
                    "`lintrunner -a` to regenerate it."
                ),
            }
            print(json.dumps(lint_message), flush=True)
        return 0
    if args.check:
        if current != generated:
            print("CODEOWNERS is out of date; run scripts/codeowners/generate_codeowners.py")
            return 1
        return 0
    codeowners.write_text(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
