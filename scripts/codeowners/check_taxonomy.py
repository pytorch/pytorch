#!/usr/bin/env python3
"""Check that the proposed CODEOWNERS review surfaces cover every tracked path."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


LINTER_CODE = "CODEOWNERS_TAXONOMY"
HIERARCHY_MARKER = "# Review-surface hierarchy:"
SECTION_MARKER = "# Draft grouped review-surface definitions:"
GROUP_PREFIX = "# ["
TaxonomyPattern = tuple[str, str]


@dataclass
class Report:
    tracked: int
    pattern_count: int
    groups: set[str]
    uncovered: list[str]
    overridden: dict[str, list[TaxonomyPattern]]
    duplicates: dict[str, list[TaxonomyPattern]]
    stale: list[TaxonomyPattern]
    ineffective: list[TaxonomyPattern]
    activation_mismatches: list[str]


def parse_declared_groups(codeowners: Path) -> list[str]:
    """Parse the review surfaces declared by the hierarchy."""
    lines = codeowners.read_text().splitlines()
    try:
        start = lines.index(HIERARCHY_MARKER) + 1
        end = lines.index(SECTION_MARKER)
    except ValueError as error:
        raise ValueError("missing review-surface hierarchy or rules marker") from error

    groups = []
    for line in lines[start:end]:
        if line.startswith("# - ") and ": " in line:
            groups.extend(line.split(": ", 1)[1].split(", "))
    duplicates = sorted(group for group in set(groups) if groups.count(group) > 1)
    if duplicates:
        raise ValueError(f"duplicate hierarchy surfaces: {', '.join(duplicates)}")
    if not groups:
        raise ValueError("review-surface hierarchy contains no surfaces")
    return groups


def parse_patterns(codeowners: Path) -> tuple[list[TaxonomyPattern], list[str]]:
    """Parse the grouped review-surface definitions from CODEOWNERS."""
    lines = codeowners.read_text().splitlines()
    try:
        start = lines.index(SECTION_MARKER) + 1
    except ValueError as error:
        raise ValueError(f"missing taxonomy marker: {SECTION_MARKER}") from error

    patterns = []
    section_groups = []
    group = None
    for line_number, raw_line in enumerate(lines[start:], start=start + 1):
        line = raw_line.strip()
        if line.startswith(GROUP_PREFIX) and line.endswith("]"):
            group = line[len(GROUP_PREFIX) : -1]
            if group in section_groups:
                raise ValueError(f"{codeowners}:{line_number}: repeated group: {group}")
            section_groups.append(group)
        elif line.startswith(("# /", "/")):
            location = f"{codeowners}:{line_number}"
            active = line.startswith("/")
            fields = (line if active else line[2:]).split()
            pattern = fields[0]
            if group is None:
                raise ValueError(f"{location}: pattern has no group")
            if active and len(fields) == 1:
                raise ValueError(f"{location}: active pattern has no owners")
            if pattern == "/" or any(character in pattern for character in "*?[\\"):
                raise ValueError(f"{location}: invalid taxonomy path: {pattern!r}")
            patterns.append((group, pattern[1:]))
        elif line and not line.startswith("#"):
            raise ValueError(f"{codeowners}:{line_number}: invalid taxonomy entry")

    if not patterns:
        raise ValueError("taxonomy section contains no patterns")
    return patterns, section_groups


def tracked_paths(repo: Path) -> list[str]:
    """List paths tracked by Git, including submodule entries."""
    output = subprocess.run(
        ["git", "-C", repo, "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    return sorted(path for path in output.split("\0") if path)


def analyze(paths: list[str], patterns: list[TaxonomyPattern]) -> Report:
    """Resolve the most-specific surface and measure coverage and redundant rules."""
    patterns_by_path: dict[str, list[TaxonomyPattern]] = defaultdict(list)
    for pattern in patterns:
        patterns_by_path[pattern[1]].append(pattern)

    activation_order = sorted(
        patterns,
        key=lambda pattern: (
            pattern[1].rstrip("/").count("/"),
            len(pattern[1]),
            pattern[1],
        ),
    )
    activation_rank = {pattern: index for index, pattern in enumerate(activation_order)}
    matched_patterns: set[TaxonomyPattern] = set()
    effective_patterns: set[TaxonomyPattern] = set()
    effective_groups = set()
    activation_mismatches = []
    uncovered = []
    overridden = {}
    for path in paths:
        parts = path.split("/")
        candidates = [path]
        candidates.extend(
            "/".join(parts[:index]) + "/" for index in range(1, len(parts))
        )
        matches = []
        for candidate in candidates:
            matches.extend(patterns_by_path.get(candidate, []))
        if not matches:
            uncovered.append(path)
            continue
        if len(matches) > 1:
            overridden[path] = matches
        matched_patterns.update(matches)
        effective_pattern = max(matches, key=lambda pattern: len(pattern[1]))
        effective_patterns.add(effective_pattern)
        effective_groups.add(effective_pattern[0])
        active_pattern = max(matches, key=activation_rank.__getitem__)
        if effective_pattern[0] != active_pattern[0]:
            activation_mismatches.append(path)

    duplicates = {
        path: entries for path, entries in patterns_by_path.items() if len(entries) > 1
    }
    return Report(
        tracked=len(paths),
        pattern_count=len(patterns),
        groups=effective_groups,
        uncovered=uncovered,
        overridden=overridden,
        duplicates=duplicates,
        stale=[pattern for pattern in patterns if pattern not in matched_patterns],
        ineffective=[
            pattern
            for pattern in patterns
            if pattern in matched_patterns and pattern not in effective_patterns
        ],
        activation_mismatches=activation_mismatches,
    )


def print_report(report: Report) -> None:
    """Print the coverage summary and each integrity failure."""
    covered = report.tracked - len(report.uncovered)
    percentage = 100 * covered / report.tracked if report.tracked else 100.0
    print(f"Tracked paths: {report.tracked}")
    print(f"Covered paths: {covered}")
    print(f"Uncovered paths: {len(report.uncovered)}")
    print(f"Coverage: {percentage:.4f}%")
    print(f"Review surfaces: {len(report.groups)}")
    print(f"Patterns: {report.pattern_count}")
    print(f"Paths with overrides: {len(report.overridden)}")
    print(f"Duplicate patterns: {len(report.duplicates)}")
    print(f"Stale patterns: {len(report.stale)}")
    print(f"Ineffective patterns: {len(report.ineffective)}")
    print(f"Generated-order mismatches: {len(report.activation_mismatches)}")

    duplicate_entries = {
        path: ", ".join(group for group, _ in patterns)
        for path, patterns in report.duplicates.items()
    }
    sections = [
        ("Uncovered paths", report.uncovered),
        (
            "Duplicate patterns",
            [f"{path}: {groups}" for path, groups in duplicate_entries.items()],
        ),
        ("Stale patterns", [f"{group}:{path}" for group, path in report.stale]),
        (
            "Ineffective patterns",
            [f"{group}:{path}" for group, path in report.ineffective],
        ),
        ("Generated-order mismatches", report.activation_mismatches),
    ]
    for title, entries in sections:
        if entries:
            print(f"\n{title}:")
            print("\n".join(f"  {entry}" for entry in entries))


def emit_lint_error(description: str) -> None:
    """Emit a lintrunner-compatible taxonomy error."""
    print(
        json.dumps(
            {
                "path": "CODEOWNERS",
                "line": None,
                "char": None,
                "code": LINTER_CODE,
                "severity": "error",
                "name": "invalid-taxonomy",
                "original": None,
                "replacement": None,
                "description": description,
            }
        ),
        flush=True,
    )


def main() -> int:
    """Run the taxonomy coverage check for the current Git repository."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lintrunner", action="store_true")
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
        codeowners = repo / "CODEOWNERS"
        declared_groups = parse_declared_groups(codeowners)
        patterns, section_groups = parse_patterns(codeowners)
        report = analyze(tracked_paths(repo), patterns)
    except ValueError as error:
        if args.lintrunner:
            emit_lint_error(
                f"CODEOWNERS taxonomy validation failed: {error}. "
                "Run `python scripts/codeowners/check_taxonomy.py` for details."
            )
            return 0
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not args.lintrunner:
        print_report(report)
    declared_group_set = set(declared_groups)
    missing_groups = sorted(declared_group_set - report.groups)
    undeclared_groups = sorted(report.groups - declared_group_set)
    if not args.lintrunner:
        if missing_groups:
            print(f"Missing declared surfaces: {', '.join(missing_groups)}")
        if undeclared_groups:
            print(f"Undeclared effective surfaces: {', '.join(undeclared_groups)}")
    section_order_mismatch = section_groups != declared_groups
    if section_order_mismatch and not args.lintrunner:
        print("Grouped sections do not match the declared hierarchy order")
    has_errors = bool(
        missing_groups or undeclared_groups or section_order_mismatch
    ) or any(
        (
            report.uncovered,
            report.duplicates,
            report.stale,
            report.ineffective,
            report.activation_mismatches,
        )
    )
    if args.lintrunner and has_errors:
        failure_counts = [
            (len(report.uncovered), "uncovered paths"),
            (len(report.duplicates), "duplicate patterns"),
            (len(report.stale), "stale patterns"),
            (len(report.ineffective), "ineffective patterns"),
            (len(report.activation_mismatches), "ordering mismatches"),
            (len(missing_groups), "missing surfaces"),
            (len(undeclared_groups), "undeclared surfaces"),
        ]
        failures = [f"{count} {name}" for count, name in failure_counts if count]
        if section_order_mismatch:
            failures.append("section order does not match the hierarchy")
        emit_lint_error(
            "CODEOWNERS taxonomy validation failed: "
            + ", ".join(failures)
            + ". Run `python scripts/codeowners/check_taxonomy.py` for details."
        )
        return 0
    return int(has_errors)


if __name__ == "__main__":
    sys.exit(main())
