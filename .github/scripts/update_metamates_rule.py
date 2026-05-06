#!/usr/bin/env python3
"""
Update the "Metamates" rule in merge_rules.yaml.

Fetches the pytorch/metamates team from GitHub, cross-references with
PR approvers from the last 90 days, and updates the "Metamates" rule
with active members not already covered by other wildcard ('*') rules.

Uses PyYAML for parsing but preserves original file formatting by doing
targeted string replacement for the approved_by list.

Requires: `gh` CLI authenticated with pytorch org access, pyyaml.
"""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml


MIN_APPROVALS = 3
SINCE = "90 days ago"
MERGE_RULES = Path(__file__).resolve().parent.parent / "merge_rules.yaml"
METAMATES_RULE = "Metamates"


def fetch_metamates() -> set[str]:
    """Fetch pytorch/metamates team members via GitHub API."""
    res = subprocess.run(
        [
            "gh",
            "api",
            "orgs/pytorch/teams/metamates/members",
            "--paginate",
            "--jq",
            ".[].login",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(res.stdout.strip().split("\n"))


def fetch_approvers() -> dict[str, int]:
    """Extract approvers and their counts from recent commit history."""
    res = subprocess.run(
        ["git", "log", f"--since={SINCE}", "--format=%b"],
        capture_output=True,
        text=True,
        check=True,
    )
    counts: dict[str, int] = defaultdict(int)
    for line in res.stdout.split("\n"):
        if "Approved by:" not in line:
            continue
        line = line.replace("https://github.com/", "").replace("Approved by:", "")
        for name in line.split(","):
            name = name.strip()
            if name:
                counts[name] += 1
    return dict(counts)


def collect_other_wildcard_approvers(rules: list[dict]) -> set[str]:
    """Collect all approvers from wildcard ('*') rules other than Metamates."""
    covered: set[str] = set()
    for rule in rules:
        if rule.get("name") == METAMATES_RULE:
            continue
        patterns = rule.get("patterns", [])
        if "*" not in patterns:
            continue
        for user in rule.get("approved_by", []):
            covered.add(str(user).lower())
    return covered


def update_approved_by(content: str, new_members: list[str]) -> str:
    """Replace the approved_by list in the Metamates rule, preserving formatting."""
    lines = content.split("\n")
    in_metamates = False
    in_approved_by = False
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"- name: {METAMATES_RULE}":
            in_metamates = True
            continue
        if in_metamates and stripped.startswith("- name:"):
            in_metamates = False
            continue
        if in_metamates and stripped == "approved_by:":
            in_approved_by = True
            start_idx = i + 1
            continue
        if in_approved_by:
            if stripped.startswith("- "):
                end_idx = i + 1
            else:
                break

    if start_idx is None or end_idx is None:
        print(
            f"ERROR: Could not find approved_by in '{METAMATES_RULE}' rule",
            file=sys.stderr,
        )
        sys.exit(1)

    new_lines = [f"  - {m}" for m in new_members]
    return "\n".join(lines[:start_idx] + new_lines + lines[end_idx:])


def main() -> None:
    content = MERGE_RULES.read_text()
    rules = yaml.safe_load(content)

    if not any(r.get("name") == METAMATES_RULE for r in rules):
        print(
            f"ERROR: No rule named '{METAMATES_RULE}' in {MERGE_RULES}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Fetching pytorch/metamates team members...")
    metamates = fetch_metamates()
    print(f"  {len(metamates)} members")

    print(f"Extracting approvers from last {SINCE}...")
    approvers = fetch_approvers()
    print(f"  {len(approvers)} unique approvers")

    covered = collect_other_wildcard_approvers(rules)
    print(f"  {len(covered)} users already in other wildcard rules")

    metamates_lower = {m.lower(): m for m in metamates}
    active = []
    for approver, count in approvers.items():
        if count < MIN_APPROVALS:
            continue
        lower = approver.lower()
        if lower in covered or lower == "pytorchbot":
            continue
        if lower in metamates_lower:
            active.append((metamates_lower[lower], count))

    members = sorted([name for name, _ in active], key=str.lower)
    print(f"  {len(members)} active metamates for '{METAMATES_RULE}' rule")

    new_content = update_approved_by(content, members)
    MERGE_RULES.write_text(new_content)
    print(f"\nUpdated {MERGE_RULES}")


if __name__ == "__main__":
    main()
