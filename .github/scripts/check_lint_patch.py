#!/usr/bin/env python3
"""Check the patch from apply-lint's lint job before mergebot applies it.

The lint job runs the PR's linters without credentials, so the patch it uploads
is as untrusted as the PR. Accept only content edits to regular files that the
PR already changes, outside .github/. Git parses the patch, the same way
`git apply` will.

Usage: check_lint_patch.py PATCH BASE_REF, run in the PR checkout.
"""

from __future__ import annotations

import subprocess
import sys


REGULAR_FILE_MODES = ("100644", "100755")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def patch_paths(numstat: str) -> list[str]:
    """Paths from `git apply --numstat -z`; a rename lists only its new path."""
    return [record.split("\t", 2)[2] for record in numstat.split("\0") if record]


def patch_problems(
    paths: list[str], summary: str, allowed: set[str], modes: dict[str, str]
) -> list[str]:
    problems = []
    if summary.strip():
        problems.append(f"creates, deletes, renames or changes modes:\n{summary}")
    for path in paths:
        if path.startswith(".github/"):
            problems.append(f"{path}: under .github/")
        elif path not in allowed:
            problems.append(f"{path}: not changed by the PR")
        elif modes.get(path) not in REGULAR_FILE_MODES:
            problems.append(f"{path}: not a regular file")
    return problems


def main(patch: str, base_ref: str) -> int:
    merge_base = git("merge-base", base_ref, "HEAD").strip()
    allowed = set(git("diff", "--name-only", "-z", merge_base, "HEAD").split("\0"))
    paths = patch_paths(git("apply", "--numstat", "-z", patch))
    modes = {}
    for line in git("ls-files", "-s", "-z", "--", *paths).split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            modes[path] = meta.split()[0]
    problems = patch_problems(paths, git("apply", "--summary", patch), allowed, modes)
    for problem in problems:
        print(f"Rejected lint patch: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
