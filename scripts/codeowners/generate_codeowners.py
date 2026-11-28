#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyyaml==6.0.2",
# ]
# ///
"""Generate CODEOWNERS and the label_parents rules of .github/label_to_label.yml from OWNERSHIP.yaml.

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
TOP_LEVEL_KEYS = {
    "planned_labels",
    "label_parents",
    "sections",
    "section_notes",
    "ownership",
}
GROUP_KEYS = {"name", "labels", "owners", "section", "paths"}
GITHUB_LABELS = "tools/linter/adapters/pytorch_labels.json"
LABEL_RULES = ".github/label_to_label.yml"
LABEL_RULES_BEGIN = "# BEGIN label_parents rules, generated from OWNERSHIP.yaml by scripts/codeowners/generate_codeowners.py"
LABEL_RULES_END = "# END label_parents rules"


@dataclass(frozen=True)
class PathRule:
    path: str
    # Handles GitHub asks for review; empty for a commented (quiet) rule.
    notify: tuple[str, ...]


@dataclass(frozen=True)
class LabelGroup:
    """Paths owned together under one label set (empty: not owned yet)."""

    # Stable identifier, independent of label renames.
    name: str
    labels: tuple[str, ...]
    owners: tuple[str, ...]
    # CODEOWNERS section the group's rules are written under (kept for owners.py).
    section: str
    paths: tuple[PathRule, ...]


@dataclass(frozen=True)
class OwnershipFile:
    planned_labels: tuple[str, ...]
    # Child label -> parent label; the label bot adds the parent to issues.
    label_parents: dict[str, str]
    sections: tuple[str, ...]
    section_notes: dict[str, tuple[str, ...]]
    groups: tuple[LabelGroup, ...]


def string_list(value, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{where} must be a list of non-empty strings, got {value!r}")
    return tuple(value)


def mapping(value, *, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping, got {value!r}")
    return value


def parse_path_rule(entry, *, where: str) -> PathRule:
    if isinstance(entry, str):
        path, notify = entry, ()
    elif isinstance(entry, dict) and len(entry) == 1:
        ((path, handles),) = entry.items()
        notify = string_list(handles, where=f"{where} {path}")
        if not notify:
            raise ValueError(f"{where} {path}: write a quiet path as a bare string")
    else:
        raise ValueError(
            f"{where}: expected '/path' or '/path: [handles]', got {entry!r}"
        )
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"{where}: path must start with '/', got {path!r}")
    return PathRule(path, notify)


def parse_group(entry, *, sections: tuple[str, ...]) -> LabelGroup:
    entry = mapping(entry, where="ownership entry")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"every ownership entry needs a `name`, got {entry!r}")
    where = f"ownership entry {name!r}"
    unknown = set(entry) - GROUP_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    missing = {"owners", "section", "paths"} - set(entry)
    if missing:
        raise ValueError(f"{where}: missing keys {sorted(missing)}")
    labels = string_list(entry.get("labels", []), where=f"{where} labels")
    owners = string_list(entry["owners"], where=f"{where} owners")
    if owners and not labels:
        raise ValueError(f"{where}: a group without labels can't have owners")
    if entry["section"] not in sections:
        raise ValueError(f"{where}: section {entry['section']!r} is not in `sections`")
    if not isinstance(entry["paths"], list) or not entry["paths"]:
        raise ValueError(f"{where}: paths must be a non-empty list")
    return LabelGroup(
        name=name,
        labels=labels,
        owners=owners,
        section=entry["section"],
        paths=tuple(parse_path_rule(rule, where=where) for rule in entry["paths"]),
    )


def parse_label_parents(
    value, *, github_labels: set[str], planned_labels: tuple[str, ...]
) -> dict[str, str]:
    parents = mapping(value, where="label_parents")
    for child, parent in parents.items():
        if not all(isinstance(label, str) and label for label in (child, parent)):
            raise ValueError(
                f"label_parents: expected 'label: parent', got {child!r}: {parent!r}"
            )
        if child not in github_labels and child not in planned_labels:
            raise ValueError(
                f"label_parents: {child!r} is not in {GITHUB_LABELS}; add labels not "
                "yet on GitHub to planned_labels"
            )
        # The bot adding a missing label would create it on GitHub.
        if parent not in github_labels:
            raise ValueError(
                f"label_parents: parent {parent!r} is not in {GITHUB_LABELS}"
            )
        seen, ancestor = {child}, parent
        while ancestor in parents:
            if ancestor in seen:
                raise ValueError(f"label_parents: cycle through {ancestor!r}")
            seen.add(ancestor)
            ancestor = parents[ancestor]
    return parents


def check_consistency(ownership: OwnershipFile) -> None:
    """Cross-entry rules: unique names and label sets."""
    names = [group.name for group in ownership.groups]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate OWNERSHIP.yaml names: {', '.join(duplicates)}")
    label_sets = {}
    for group in ownership.groups:
        if group.labels:
            key = frozenset(group.labels)
            if key in label_sets:
                raise ValueError(
                    f"entries {label_sets[key]!r} and {group.name!r} have the same labels"
                )
            label_sets[key] = group.name


def load_ownership(path: Path, github_labels: set[str]) -> OwnershipFile:
    """Load and validate OWNERSHIP.yaml; every problem is a ValueError."""
    try:
        document = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"not valid YAML: {error}") from error
    document = mapping(document, where="OWNERSHIP.yaml")
    if set(document) != TOP_LEVEL_KEYS:
        raise ValueError(
            f"top-level keys must be {sorted(TOP_LEVEL_KEYS)}, got {sorted(document)}"
        )
    sections = string_list(document["sections"], where="sections")
    section_notes = {
        section: string_list(notes, where=f"section_notes {section}")
        for section, notes in mapping(
            document["section_notes"], where="section_notes"
        ).items()
    }
    unknown_notes = sorted(set(section_notes) - set(sections))
    if unknown_notes:
        raise ValueError(
            f"section_notes for sections not in `sections`: {unknown_notes}"
        )
    if not isinstance(document["ownership"], list):
        raise ValueError("ownership must be a list")
    planned_labels = string_list(document["planned_labels"], where="planned_labels")
    ownership = OwnershipFile(
        planned_labels=planned_labels,
        label_parents=parse_label_parents(
            document["label_parents"],
            github_labels=github_labels,
            planned_labels=planned_labels,
        ),
        sections=sections,
        section_notes=section_notes,
        groups=tuple(
            parse_group(entry, sections=sections) for entry in document["ownership"]
        ),
    )
    check_consistency(ownership)
    return ownership


def handle(name: str) -> str:
    """Handles are stored without "@" (as in merge_rules.yaml); emails keep theirs."""
    return name if "@" in name else f"@{name}"


def render_rule(rule: PathRule) -> str:
    if not rule.notify:
        return f"# {rule.path}"
    return " ".join([rule.path, *map(handle, rule.notify)])


def render_header(ownership: OwnershipFile) -> list[str]:
    planned = (
        [
            "# - Labels planned but not yet created on GitHub (issues can't be routed to",
            f"#   them until they exist): {', '.join(ownership.planned_labels)}",
        ]
        if ownership.planned_labels
        else []
    )
    return [
        "# DO NOT MODIFY THIS FILE DIRECTLY!! This file is generated from OWNERSHIP.yaml by",
        "# scripts/codeowners/generate_codeowners.py. Edit OWNERSHIP.yaml and rerun",
        "# the script (or `spin fixlint`) instead of editing this file.",
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
        '# - The "# owner:" line directly below a "# label:" line records the group',
        "#   meant to own every path in that label block. These lines are",
        "#   informational: nothing consumes them yet, and they don't change who is",
        '#   notified. "none" means no owning group is recorded yet. Everyone an active',
        "#   rule in the block notifies must also be on that owner line, apart from the",
        "#   exceptions listed in scripts/codeowners/check_taxonomy.py.",
        "# - Sections are ordered from broad defaults to narrower overrides.",
        '# - A "# label:" line names the GitHub issue/PR labels for the paths below',
        '#   it, up to the next "# label:" line or the end of the section. Paths',
        '#   before the first "# label:" line in a section have no single label.',
        "# - A label list identifies one ownership area, so each label list (compared",
        '#   as a set) appears on exactly one "# label:" line. When a list extends a',
        "#   sibling's, the shared labels come first and the distinguishing label last.",
        "# - A label line leaves out an oncall label when one of its module labels",
        "#   implies it; label_parents in OWNERSHIP.yaml records which labels imply",
        "#   which.",
        *planned,
        "#",
        "# For module labels => owners mapping, please see https://github.com/pytorch/pytorch/issues/24422.",
        "",
        "# CODEOWNERS review-surface taxonomy",
        "#",
        "# Every tracked path resolves to one review surface. Every listed pattern must",
        "# match at least one committed path. The fast lintrunner set enforces these",
        "# rules, declared sections, valid owners, source order, no duplicate or",
        "# ineffective patterns, and that an active rule's handles appear on its",
        "# block's owner line.",
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
    """True if every path matching `narrow` also matches `broad`.

    Only literal patterns are compared: a directory (`/a/`) or a path without
    wildcards (`/a/b`, which also covers `/a/b/...`). Patterns with `*` are
    left to the taxonomy lint, which checks the generated order against the
    committed tree.
    """
    if "*" in broad or narrow == broad:
        return False
    prefix = broad if broad.endswith("/") else broad + "/"
    return narrow.startswith(prefix)


def order_blocks(groups: list[LabelGroup]) -> list[LabelGroup]:
    """Order a section's label blocks so broader patterns come before narrower ones.

    GitHub applies the last matching rule, so a block containing a directory that
    another block's patterns sit inside must come first. Otherwise blocks keep
    their OWNERSHIP.yaml order.
    """

    def precedes(first: LabelGroup, second: LabelGroup) -> bool:
        return any(contains(a.path, b.path) for a in first.paths for b in second.paths)

    remaining = list(groups)
    ordered = []
    while remaining:
        ready = next(
            (
                group
                for group in remaining
                if not any(
                    precedes(other, group) for other in remaining if other is not group
                )
            ),
            None,
        )
        if ready is None:
            names = ", ".join(group.name for group in remaining)
            raise ValueError(f"label groups nest inside each other both ways: {names}")
        ordered.append(ready)
        remaining.remove(ready)
    return ordered


def check_unlabeled_order(groups: list[LabelGroup]) -> None:
    """Paths without labels are written first in a section, so none may sit inside a labeled pattern."""
    labeled = [group for group in groups if group.labels]
    for group in groups:
        if group.labels:
            continue
        for rule in group.paths:
            for other in labeled:
                broader = next(
                    (r for r in other.paths if contains(r.path, rule.path)), None
                )
                if broader:
                    raise ValueError(
                        f"{rule.path} in {group.name!r} sits inside {broader.path} in "
                        f"{other.name!r}, but paths without labels are written first in "
                        "their section; give it a label group or another section"
                    )


def render_codeowners(ownership: OwnershipFile) -> str:
    lines = render_header(ownership)
    for section in ownership.sections:
        lines.append(f"# [{section}]")
        lines.extend(ownership.section_notes.get(section, ()))
        groups = [group for group in ownership.groups if group.section == section]
        check_unlabeled_order(groups)
        for group in groups:
            if not group.labels:
                lines.extend(map(render_rule, group.paths))
        for group in order_blocks([group for group in groups if group.labels]):
            lines.append(f"# label: {', '.join(group.labels)}")
            lines.append(f"# owner: {' '.join(map(handle, group.owners)) or 'none'}")
            lines.extend(map(render_rule, group.paths))
        lines.append("#")
    return "\n".join(lines) + "\n"


def splice_label_rules(current: str, ownership: OwnershipFile) -> str:
    """Replace the generated block of label_to_label.yml with one rule per parent."""
    children: dict[str, list[str]] = {}
    for child, parent in ownership.label_parents.items():
        children.setdefault(parent, []).append(child)
    rules = []
    for parent, labels in children.items():
        rules += ["- any:", *(f"  - {json.dumps(label)}" for label in labels)]
        rules += ["  then:", f"  - {json.dumps(parent)}"]
    lines = current.splitlines()
    begins = [i for i, line in enumerate(lines) if line == LABEL_RULES_BEGIN]
    ends = [i for i, line in enumerate(lines) if line == LABEL_RULES_END]
    if len(begins) != 1 or len(ends) != 1 or begins[0] > ends[0]:
        raise ValueError(
            f"{LABEL_RULES} needs one line {LABEL_RULES_BEGIN!r} followed by one line "
            f"{LABEL_RULES_END!r}"
        )
    begin, end = begins[0], ends[0]
    return "\n".join([*lines[: begin + 1], *rules, *lines[end:]]) + "\n"


def emit_lint_error(
    *, path: str, description: str, original: str | None, replacement: str | None
) -> None:
    """Emit a lintrunner message; with a replacement, `lintrunner -a` applies it."""
    lint_message = {
        "path": path,
        "line": None,
        "char": None,
        "code": LINTER_CODE,
        "severity": "error",
        "name": "stale-codeowners" if replacement is not None else "invalid-ownership",
        "original": original,
        "replacement": replacement,
        "description": description,
    }
    print(json.dumps(lint_message), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="fail if a generated file is stale"
    )
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
    try:
        github_labels = set(json.loads((repo / GITHUB_LABELS).read_text()))
        ownership = load_ownership(repo / "OWNERSHIP.yaml", github_labels)
        outputs = ("CODEOWNERS", LABEL_RULES)
        current = {name: (repo / name).read_text() for name in outputs}
        generated = {
            "CODEOWNERS": render_codeowners(ownership),
            LABEL_RULES: splice_label_rules(current[LABEL_RULES], ownership),
        }
    except ValueError as error:
        if args.lintrunner:
            emit_lint_error(
                path="OWNERSHIP.yaml",
                description=f"OWNERSHIP.yaml is invalid: {error}",
                original=None,
                replacement=None,
            )
            return 0
        print(f"error: OWNERSHIP.yaml is invalid: {error}", file=sys.stderr)
        return 1
    stale = [name for name in generated if current[name] != generated[name]]
    if args.lintrunner:
        for name in stale:
            emit_lint_error(
                path=name,
                description=(
                    f"{name} is generated from OWNERSHIP.yaml, so direct edits to its "
                    "generated part are discarded. Make the change in OWNERSHIP.yaml, "
                    "then run `spin fixlint` or "
                    "`python scripts/codeowners/generate_codeowners.py` to regenerate it."
                ),
                original=current[name],
                replacement=generated[name],
            )
        return 0
    if args.check:
        if stale:
            print(
                f"Out of date: {', '.join(stale)}. Direct edits to generated content "
                "are discarded: make the change in OWNERSHIP.yaml, then run "
                "scripts/codeowners/generate_codeowners.py"
            )
            return 1
        return 0
    for name in stale:
        (repo / name).write_text(generated[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
