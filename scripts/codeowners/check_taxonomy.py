#!/usr/bin/env python3
"""Check that the proposed CODEOWNERS review surfaces cover every tracked path."""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


LINTER_CODE = "CODEOWNERS_TAXONOMY"
HIERARCHY_MARKER = "# Review-surface hierarchy:"
LABEL_PREFIX = "# label:"
OWNER_PREFIX = "# owner:"
SECTION_MARKER = "# Grouped review-surface definitions:"
GROUP_PREFIX = "# ["

FAILURE_GUIDANCE = {
    "Uncovered paths": (
        "In OWNERSHIP.yaml, add the path to an entry in the right section: a bare "
        "path if owners are unknown, or `/path/: [handle]` to notify someone."
    ),
    "Duplicate patterns": (
        "Search OWNERSHIP.yaml for the exact pattern and keep it in only one entry."
    ),
    "Stale patterns": (
        "In OWNERSHIP.yaml, correct or remove the reported pattern; it matches no "
        "committed path."
    ),
    "Ineffective patterns": (
        "In OWNERSHIP.yaml, move the reported pattern to the entry that wins for its "
        "paths, narrow it, or remove it."
    ),
    "Source-order mismatches": (
        "In OWNERSHIP.yaml, put the broader pattern in an earlier section than the "
        "narrower one; GitHub applies the last matching rule."
    ),
    "Missing effective surfaces": (
        "In OWNERSHIP.yaml, give the named section an entry with a path, or remove it "
        "from `sections`."
    ),
    "Missing grouped sections": (
        "In OWNERSHIP.yaml, add an entry for the missing section or remove it from "
        "`sections`."
    ),
    "Undeclared grouped sections": (
        "In OWNERSHIP.yaml, add the section to `sections` or move its entries to "
        "another section."
    ),
    "Invalid owner handles": (
        "In OWNERSHIP.yaml, list GitHub users, teams, or emails in the entry's "
        "`owners`."
    ),
    "Reviewers missing from owner lists": (
        "In OWNERSHIP.yaml, add the reported handles to the entry's `owners`, or "
        "move the path to an entry whose owners include them."
    ),
    "Unneeded reviewer exceptions": (
        "Remove the reported entries from REVIEWERS_NOT_OWNERS in "
        "scripts/codeowners/check_taxonomy.py; their handles are owners now, no "
        "longer on the rule, or the rule is gone."
    ),
}

# Active rules whose handles are notified without being owners of the rule's label
# block. These predate the owner lines and are left as they are until the people
# involved decide; every other rule's handles must be on its block's owner line.
REVIEWERS_NOT_OWNERS = {
    "test/functorch/test_aotdispatch.py": frozenset({"@chillee", "@ezyang"}),
    "torch/_export/serde/schema.py": frozenset({"@sherlocknomad"}),
    "torch/fx/proxy.py": frozenset({"@zou3519"}),
}


@dataclass(frozen=True)
class TaxonomyPattern:
    """A grouped taxonomy rule at its actual CODEOWNERS source position."""

    group: str
    path: str
    line_number: int
    labels: tuple[str, ...] = ()
    # Handles on the block's `# owner:` line; None when the block has no owner line.
    owners: tuple[str, ...] | None = None
    # Handles on the rule itself, who GitHub asks for review.
    reviewers: tuple[str, ...] = ()


@dataclass(frozen=True)
class OwnerLine:
    line_number: int
    text: str


@dataclass(frozen=True)
class ParsedCodeowners:
    patterns: list[TaxonomyPattern]
    section_groups: list[str]
    owner_lines: list[OwnerLine]


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
    source_order_mismatches: list[str]


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


def is_valid_owner(owner: str) -> bool:
    """Check the syntax of a GitHub user, team, or email CODEOWNER."""
    if owner.startswith("@"):
        parts = owner[1:].split("/")
        return len(parts) <= 2 and all(
            part
            and all(character.isalnum() or character in "-_." for character in part)
            for part in parts
        )
    if owner.count("@") != 1:
        return False
    local, domain = owner.split("@")
    return bool(local and domain)


def parse_patterns(codeowners: Path) -> tuple[list[TaxonomyPattern], list[str]]:
    """Parse grouped rules while preserving their actual CODEOWNERS order."""
    parsed = parse_codeowners(codeowners)
    return parsed.patterns, parsed.section_groups


def owner_handles(owner_line: str) -> tuple[str, ...]:
    """Handles on an `# owner:` line; `none` means an empty list."""
    handles = tuple(owner_line.removeprefix(OWNER_PREFIX).split())
    return () if handles == ("none",) else handles


def has_valid_handles(owner_line: str) -> bool:
    """An `# owner:` line names valid handles, or `none` on its own."""
    handles = owner_line.removeprefix(OWNER_PREFIX).split()
    return handles == ["none"] or (bool(handles) and all(map(is_valid_owner, handles)))


def parse_codeowners(codeowners: Path) -> ParsedCodeowners:
    """Parse grouped rules, label blocks and owner lines in one pass."""
    lines = codeowners.read_text().splitlines()
    try:
        marker = lines.index(SECTION_MARKER)
    except ValueError as error:
        raise ValueError(f"missing taxonomy marker: {SECTION_MARKER}") from error
    for line_number, raw_line in enumerate(lines[:marker], start=1):
        line = raw_line.strip()
        if line and not line.startswith("#"):
            raise ValueError(
                f"{codeowners}:{line_number}: active rule outside grouped taxonomy"
            )

    start = marker + 1
    patterns = []
    section_groups = []
    owner_lines = []
    group = None
    labels: tuple[str, ...] = ()
    block_owners: tuple[str, ...] | None = None
    for line_number, raw_line in enumerate(lines[start:], start=start + 1):
        line = raw_line.strip()
        if line.startswith(GROUP_PREFIX) and line.endswith("]"):
            group = line[len(GROUP_PREFIX) : -1]
            labels = ()
            block_owners = None
            if group in section_groups:
                raise ValueError(f"{codeowners}:{line_number}: repeated group: {group}")
            section_groups.append(group)
        elif line.startswith(LABEL_PREFIX):
            label_names = line.removeprefix(LABEL_PREFIX).split(",")
            labels = tuple(label.strip() for label in label_names)
            block_owners = None
        elif line.startswith(OWNER_PREFIX):
            owner_lines.append(OwnerLine(line_number, line))
            block_owners = owner_handles(line)
        elif line.startswith(("# /", "/")):
            location = f"{codeowners}:{line_number}"
            active = line.startswith("/")
            fields = (line if active else line[2:]).split()
            pattern = fields[0]
            if group is None:
                raise ValueError(f"{location}: pattern has no group")
            if active:
                owners = []
                for field in fields[1:]:
                    if field.startswith("#"):
                        break
                    owners.append(field)
                if not owners:
                    raise ValueError(f"{location}: active pattern has no owners")
                invalid_owners = [
                    owner for owner in owners if not is_valid_owner(owner)
                ]
                if invalid_owners:
                    raise ValueError(
                        f"{location}: invalid active owners: {', '.join(invalid_owners)}"
                    )
            if pattern == "/" or any(character in pattern for character in "?[\\"):
                raise ValueError(f"{location}: invalid taxonomy path: {pattern!r}")
            reviewers = tuple(owners) if active else ()
            patterns.append(
                TaxonomyPattern(
                    group, pattern[1:], line_number, labels, block_owners, reviewers
                )
            )
        elif line and not line.startswith("#"):
            raise ValueError(f"{codeowners}:{line_number}: invalid taxonomy entry")

    if not patterns:
        raise ValueError("taxonomy section contains no patterns")
    return ParsedCodeowners(patterns, section_groups, owner_lines)


def find_ownership_errors(
    parsed: ParsedCodeowners,
    *,
    reviewers_not_owners: dict[str, frozenset[str]] = REVIEWERS_NOT_OWNERS,
) -> dict[str, list[str]]:
    """Check owner handles, and that each rule's reviewers are on its block's owner line.

    The generator writes one owner line below each label line and rejects repeated
    label lists, so those are not checked here.
    """
    invalid_handles = [
        f"CODEOWNERS:{line.line_number} {line.text}"
        for line in parsed.owner_lines
        if not has_valid_handles(line.text)
    ]
    # GitHub handles are case-insensitive, so exceptions are too.
    exceptions = {
        path: {handle.casefold() for handle in handles}
        for path, handles in reviewers_not_owners.items()
    }
    missing_reviewers = []
    used_exceptions = set()
    for pattern in parsed.patterns:
        if not pattern.labels or pattern.owners is None:
            continue
        owners = {owner.casefold() for owner in pattern.owners}
        allowed = exceptions.get(pattern.path, set())
        missing = []
        for reviewer in pattern.reviewers:
            if reviewer.casefold() in owners:
                continue
            if reviewer.casefold() in allowed:
                used_exceptions.add((pattern.path, reviewer.casefold()))
            else:
                missing.append(reviewer)
        if missing:
            missing_reviewers.append(
                f"CODEOWNERS:{pattern.line_number} /{pattern.path}: "
                f"{' '.join(missing)} not on the block's owner line"
            )
    # Every exception must still be needed, including ones whose rule was removed
    # or renamed, so a stale entry can't later exempt a new rule unreviewed.
    unneeded_exceptions = [
        f"/{path}: {handle}"
        for path, handles in sorted(exceptions.items())
        for handle in sorted(handles)
        if (path, handle) not in used_exceptions
    ]
    return {
        "Invalid owner handles": invalid_handles,
        "Reviewers missing from owner lists": missing_reviewers,
        "Unneeded reviewer exceptions": unneeded_exceptions,
    }


def tracked_paths(repo: Path) -> list[str]:
    """List paths committed in HEAD, including submodule entries."""
    output = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "-z", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    return sorted(path for path in output.split("\0") if path)


@lru_cache
def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile the CODEOWNERS glob subset used by this repository."""
    expression = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            expression.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            raise ValueError(f"unsupported CODEOWNERS glob: {pattern!r}")
        elif pattern[index] == "*":
            expression.append("[^/]*")
            index += 1
        else:
            expression.append(re.escape(pattern[index]))
            index += 1
    return re.compile("^" + "".join(expression) + "(?:/.*)?$")


def pattern_specificity(pattern: TaxonomyPattern) -> tuple[int, int, int]:
    """Rank literal paths and globs by their non-wildcard prefix."""
    prefix = pattern.path.split("*", 1)[0]
    return prefix.count("/"), len(prefix), len(pattern.path)


class TaxonomyIndex:
    """Look up the taxonomy patterns that match a tracked path."""

    def __init__(self, patterns: list[TaxonomyPattern]) -> None:
        self.patterns_by_path: dict[str, list[TaxonomyPattern]] = defaultdict(list)
        self.glob_patterns: list[TaxonomyPattern] = []
        for pattern in patterns:
            if "*" in pattern.path:
                self.glob_patterns.append(pattern)
            else:
                self.patterns_by_path[pattern.path].append(pattern)

    def matches(self, path: str) -> list[TaxonomyPattern]:
        """Return the patterns matching the path or one of its parent directories."""
        parts = path.split("/")
        candidates = [path]
        candidates.extend(
            "/".join(parts[:index]) + "/" for index in range(1, len(parts))
        )
        matches: list[TaxonomyPattern] = []
        for candidate in candidates:
            matches.extend(self.patterns_by_path.get(candidate, []))
        matches.extend(
            pattern
            for pattern in self.glob_patterns
            if compile_glob(pattern.path).fullmatch(path)
        )
        return matches


def analyze(paths: list[str], patterns: list[TaxonomyPattern]) -> Report:
    """Measure coverage and verify that source order preserves specificity."""
    index = TaxonomyIndex(patterns)
    matched_patterns: set[TaxonomyPattern] = set()
    effective_patterns: set[TaxonomyPattern] = set()
    effective_groups = set()
    source_order_mismatches = []
    uncovered = []
    overridden = {}
    for path in paths:
        matches = index.matches(path)
        if not matches:
            uncovered.append(path)
            continue
        if len(matches) > 1:
            overridden[path] = matches
        matched_patterns.update(matches)
        effective_pattern = max(matches, key=pattern_specificity)
        effective_patterns.update(
            pattern for pattern in matches if pattern.group == effective_pattern.group
        )
        effective_groups.add(effective_pattern.group)
        source_pattern = max(matches, key=lambda pattern: pattern.line_number)
        if effective_pattern != source_pattern:
            source_order_mismatches.append(
                f"{path}: CODEOWNERS:{source_pattern.line_number} "
                f"[{source_pattern.group}] /{source_pattern.path} overrides "
                f"CODEOWNERS:{effective_pattern.line_number} "
                f"[{effective_pattern.group}] /{effective_pattern.path}"
            )

    duplicates = {
        path: entries
        for path, entries in index.patterns_by_path.items()
        if len(entries) > 1
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
        source_order_mismatches=source_order_mismatches,
    )


def print_failure_entries(title: str, entries: list[str]) -> None:
    """Print failing entries with the corresponding remediation."""
    if not entries:
        return
    print(f"\n{title} ({len(entries)}):")
    print("\n".join(f"  {entry}" for entry in entries))
    print(f"  Fix: {FAILURE_GUIDANCE[title]}")


def print_report(report: Report) -> None:
    """Print an explained coverage summary and each integrity failure."""
    covered = report.tracked - len(report.uncovered)
    percentage = 100 * covered / report.tracked if report.tracked else 100.0
    print("CODEOWNERS taxonomy summary")
    print("  CODEOWNERS is generated: fix failures in OWNERSHIP.yaml, then run")
    print(
        "  scripts/codeowners/generate_codeowners.py. Line numbers refer to CODEOWNERS."
    )
    print("  Write `/path/: [handle]` to notify someone or a bare `/path/` otherwise.")
    print(f"  Tracked paths: {report.tracked} (committed paths checked)")
    print(f"  Covered paths: {covered} (matched by at least one pattern)")
    print(f"  Coverage: {percentage:.4f}% (must remain 100%)")
    print(
        f"  Review surfaces: {len(report.groups)} "
        "(sections that effectively cover committed paths)"
    )
    print(
        f"  Patterns: {report.pattern_count} "
        "(active ownership rules and commented unresolved paths)"
    )
    print(
        f"  Paths matched by multiple patterns: {len(report.overridden)} "
        "(expected when narrower patterns follow broader ones)"
    )
    print("\nIntegrity checks (all must be zero)")
    print(f"  Uncovered paths: {len(report.uncovered)}")
    print(f"  Duplicate patterns: {len(report.duplicates)}")
    print(f"  Stale patterns: {len(report.stale)}")
    print(f"  Ineffective patterns: {len(report.ineffective)}")
    print(f"  Source-order mismatches: {len(report.source_order_mismatches)}")

    duplicate_entries = {
        path: ", ".join(
            f"CODEOWNERS:{pattern.line_number} [{pattern.group}]"
            for pattern in patterns
        )
        for path, patterns in report.duplicates.items()
    }
    sections = [
        ("Uncovered paths", report.uncovered),
        (
            "Duplicate patterns",
            [f"/{path}: {locations}" for path, locations in duplicate_entries.items()],
        ),
        (
            "Stale patterns",
            [
                f"CODEOWNERS:{pattern.line_number} [{pattern.group}] /{pattern.path}"
                for pattern in report.stale
            ],
        ),
        (
            "Ineffective patterns",
            [
                f"CODEOWNERS:{pattern.line_number} [{pattern.group}] /{pattern.path}"
                for pattern in report.ineffective
            ],
        ),
        ("Source-order mismatches", report.source_order_mismatches),
    ]
    for title, entries in sections:
        print_failure_entries(title, entries)


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
        parsed = parse_codeowners(codeowners)
        patterns = parsed.patterns
        section_groups = parsed.section_groups
        report = analyze(tracked_paths(repo), patterns)
        ownership_errors = find_ownership_errors(
            parsed, reviewers_not_owners=REVIEWERS_NOT_OWNERS
        )
    except ValueError as error:
        if args.lintrunner:
            emit_lint_error(
                f"CODEOWNERS taxonomy validation failed: {error}. "
                "Fix it in OWNERSHIP.yaml and regenerate CODEOWNERS, then run "
                "`python scripts/codeowners/check_taxonomy.py` locally."
            )
            return 0
        print(f"error: {error}", file=sys.stderr)
        return 2

    declared_group_set = set(declared_groups)
    section_group_set = set(section_groups)
    missing_groups = sorted(declared_group_set - report.groups)
    missing_sections = sorted(declared_group_set - section_group_set)
    undeclared_sections = sorted(section_group_set - declared_group_set)
    has_errors = (
        bool(missing_groups or missing_sections or undeclared_sections)
        or any(ownership_errors.values())
        or any(
            (
                report.uncovered,
                report.duplicates,
                report.stale,
                report.ineffective,
                report.source_order_mismatches,
            )
        )
    )
    if not args.lintrunner:
        print_report(report)
        for title, entries in ownership_errors.items():
            print_failure_entries(title, entries)
        print_failure_entries("Missing effective surfaces", missing_groups)
        print_failure_entries("Missing grouped sections", missing_sections)
        print_failure_entries("Undeclared grouped sections", undeclared_sections)
        print(f"\nResult: {'FAIL' if has_errors else 'PASS'}")
    if args.lintrunner and has_errors:
        failure_counts = [
            (len(report.uncovered), "uncovered paths"),
            (len(report.duplicates), "duplicate patterns"),
            (len(report.stale), "stale patterns"),
            (len(report.ineffective), "ineffective patterns"),
            (len(report.source_order_mismatches), "source-order mismatches"),
            (len(missing_groups), "missing effective surfaces"),
            (len(missing_sections), "missing grouped sections"),
            (len(undeclared_sections), "undeclared grouped sections"),
            *(
                (len(entries), title.lower())
                for title, entries in ownership_errors.items()
            ),
        ]
        failures = [f"{count} {name}" for count, name in failure_counts if count]
        emit_lint_error(
            "CODEOWNERS taxonomy validation failed: "
            + ", ".join(failures)
            + ". Run `python scripts/codeowners/check_taxonomy.py`; it lists each "
            "offending entry and how to fix it."
        )
        return 0
    return int(has_errors)


if __name__ == "__main__":
    sys.exit(main())
