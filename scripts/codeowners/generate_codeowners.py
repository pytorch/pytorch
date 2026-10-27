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


@dataclass(frozen=True)
class Area:
    labels: tuple[str, ...]
    owners: tuple[str, ...]
    paths: tuple[PathRule, ...]


@dataclass(frozen=True)
class Section:
    name: str
    note: tuple[str, ...]
    # Paths before the section's first label block.
    paths: tuple[PathRule, ...]
    areas: tuple[Area, ...]


@dataclass(frozen=True)
class Category:
    name: str
    sections: tuple[str, ...]


@dataclass(frozen=True)
class Ownership:
    oncall_queues: dict[str, tuple[str, ...]]
    planned_labels: tuple[str, ...]
    categories: tuple[Category, ...]
    sections: tuple[Section, ...]


def parse_path_rule(entry) -> PathRule:
    if isinstance(entry, str):
        return PathRule(entry, ())
    ((path, notify),) = entry.items()
    return PathRule(path, tuple(notify))


def load_ownership(path: Path) -> Ownership:
    document = yaml.safe_load(path.read_text())
    return Ownership(
        oncall_queues={
            oncall: tuple(modules)
            for oncall, modules in document["oncall_queues"].items()
        },
        planned_labels=tuple(document["planned_labels"]),
        categories=tuple(
            Category(entry["name"], tuple(entry["sections"]))
            for entry in document["categories"]
        ),
        sections=tuple(
            Section(
                name=entry["name"],
                note=tuple(entry.get("note", ())),
                paths=tuple(map(parse_path_rule, entry.get("paths", ()))),
                areas=tuple(
                    Area(
                        labels=tuple(area["labels"]),
                        owners=tuple(area["owners"]),
                        paths=tuple(map(parse_path_rule, area["paths"])),
                    )
                    for area in entry.get("areas", ())
                ),
            )
            for entry in document["sections"]
        ),
    )


def handle(name: str) -> str:
    """Store handles without "@" (as merge_rules.yaml does); emails keep theirs."""
    return name if "@" in name else f"@{name}"


def render_rule(rule: PathRule) -> str:
    if not rule.notify:
        return f"# {rule.path}"
    return " ".join([rule.path, *map(handle, rule.notify)])


def render_header(ownership: Ownership) -> list[str]:
    oncall_map = [
        f"#     {oncall}: {', '.join(m.removeprefix('module: ') for m in modules)}"
        for oncall, modules in ownership.oncall_queues.items()
    ]
    hierarchy = [
        f"# - {category.name}: {', '.join(category.sections)}"
        for category in ownership.categories
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
        "# Review-surface hierarchy:",
        *hierarchy,
        "#",
        SECTION_MARKER,
        "#",
    ]


def render_codeowners(ownership: Ownership) -> str:
    lines = render_header(ownership)
    for section in ownership.sections:
        lines.append(f"# [{section.name}]")
        lines.extend(section.note)
        lines.extend(map(render_rule, section.paths))
        for area in section.areas:
            lines.append(f"# label: {', '.join(area.labels)}")
            lines.append(f"# owner: {' '.join(map(handle, area.owners)) or 'none'}")
            lines.extend(map(render_rule, area.paths))
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
