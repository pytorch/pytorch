#!/usr/bin/env python3
"""Write the changes the author made since the latest automated review.

Runs in the review job before the model, from the trusted checkout, on the
PR's working copy (`--repo`). Its inputs are commits the trusted `prepare` job resolved
through the API: the commit the latest automated review saw, that commit's
merge base, and this run's merge base.

Diffing the earlier commit against HEAD is wrong after a rebase: changes the
base made to the PR's files read as the author's. So the earlier version is
first replayed onto this run's merge base (`git merge-tree` with an explicit
merge base, which needs no checkout and no shared history), and HEAD is diffed
against that, so upstream changes cancel out. A file the replay cannot merge
changed both upstream and in the PR; it gets the plain diff under a heading
that says it includes upstream changes. Files the earlier version changed are
included even if the PR no longer does, so a reverted file shows up. Git reads
no attributes from the PR (`--attr-source` is the empty tree), so its
`.gitattributes` cannot pick merge or diff drivers. Every other failure, and a
result over the size cap, writes a one-line note, so the review never fails for
this file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


NOTE_NO_REVIEW = "# No automated review of an earlier commit to compare with.\n"
NOTE_UNAVAILABLE = (
    "# The commit the last automated review saw, {sha}, or its merge base is no "
    "longer available.\n"
)
NOTE_NO_FILES = "# The PR changes no files.\n"
NOTE_SAME = "# The last automated review saw this same commit: nothing changed since.\n"
NOTE_FAILED = "# The changes since the last automated review could not be computed.\n"
NOTE_BASE_ONLY = (
    "# Nothing changed in this PR's files since the last automated review ({sha}) "
    "apart from changes from the base.\n"
)
NOTE_TOO_LARGE = (
    "# The changes since the last automated review are too large to list.\n"
)
HEADER = (
    "# Changes to this PR's files since the last automated review ({sha}), with "
    "the earlier version replayed onto the current merge base so that changes "
    "from the base are left out.\n"
)
CONFLICTED = (
    "\n# These files changed both in the base and in this PR since the last "
    "automated review, so the earlier version could not be replayed; their diff "
    "against the earlier commit includes the base's changes.\n"
)


REPO = Path(".")
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def git(*args: str) -> subprocess.CompletedProcess[bytes]:
    # Literal pathspecs everywhere: a PR file name is data, never pathspec magic.
    # Attributes from the empty tree: the PR's .gitattributes picks no driver.
    cmd = ["git", "-C", str(REPO), "--literal-pathspecs", f"--attr-source={EMPTY_TREE}"]
    return subprocess.run([*cmd, *args], capture_output=True)


def has_commit(sha: str) -> bool:
    return git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def replay(previous: str, previous_base: str, base: str) -> tuple[str, set[str]] | None:
    """The tree of `previous` replayed onto `base`, and the paths that conflicted."""
    proc = git(
        "merge-tree",
        "--write-tree",
        "-z",
        "--name-only",
        "--no-messages",
        "--merge-base",
        previous_base,
        base,
        previous,
    )
    # 0 is a clean merge, 1 a merge with conflicts; anything else is an error.
    if proc.returncode not in (0, 1):
        return None
    # File names are bytes; fsdecode round-trips any of them (surrogateescape).
    tree, _, rest = os.fsdecode(proc.stdout).partition("\0")
    return tree, {p for p in rest.split("\0") if p}


def diff(old: str, files: list[str]) -> str:
    if not files:
        return ""
    proc = git("diff", old, "HEAD", "--", *files)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    return proc.stdout.decode(errors="replace")


def interdiff(previous: str, previous_base: str, base: str, files: list[str]) -> str:
    if not previous:
        return NOTE_NO_REVIEW
    if not files:
        return NOTE_NO_FILES
    if git("rev-parse", "HEAD").stdout.decode().strip() == previous:
        return NOTE_SAME
    if not previous_base or not all(map(has_commit, (previous, previous_base, base))):
        return NOTE_UNAVAILABLE.format(sha=previous)
    replayed = replay(previous, previous_base, base)
    if replayed is None:
        return NOTE_FAILED
    tree, conflicted = replayed
    earlier = git("diff", "--name-only", "-z", previous_base, previous)
    if earlier.returncode != 0:
        return NOTE_FAILED
    files += [
        f for f in os.fsdecode(earlier.stdout).split("\0") if f and f not in files
    ]
    clean = diff(tree, [f for f in files if f not in conflicted])
    both = diff(previous, [f for f in files if f in conflicted])
    if not clean and not both:
        return NOTE_BASE_ONLY.format(sha=previous)
    return HEADER.format(sha=previous) + clean + (CONFLICTED + both if both else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo", type=Path, default=Path("."), help="the PR's working copy"
    )
    ap.add_argument(
        "--previous", default="", help="commit the latest automated review saw"
    )
    ap.add_argument("--previous-base", default="", help="that commit's merge base")
    ap.add_argument("--base", required=True, help="this run's merge base")
    ap.add_argument(
        "--files", required=True, type=Path, help="NUL-separated changed files"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-bytes", required=True, type=int)
    args = ap.parse_args(argv)
    global REPO
    REPO = args.repo
    files = [os.fsdecode(f) for f in args.files.read_bytes().split(b"\0") if f]
    try:
        text = interdiff(args.previous, args.previous_base, args.base, files)
    except Exception as e:  # noqa: BLE001 - any failure is a note, never a failed review
        print(f"::warning::interdiff not built: {e}")
        text = NOTE_FAILED
    if len(text.encode()) > args.max_bytes:
        text = NOTE_TOO_LARGE
    args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
