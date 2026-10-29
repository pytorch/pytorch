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
that says it includes upstream changes. Every other failure, and a result over
the size cap, writes a one-line note, so the review never fails for this file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


NOTE_NO_REVIEW = "# No automated review of an earlier commit to compare with.\n"
NOTE_UNAVAILABLE = (
    "# The commit the last automated review saw, {sha}, or its merge base is no "
    "longer available.\n"
)
NOTE_NO_FILES = "# The PR changes no files.\n"
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


def git(*args: str) -> subprocess.CompletedProcess[bytes]:
    # Literal pathspecs everywhere: a PR file name is data, never pathspec magic.
    return subprocess.run(
        ["git", "-C", str(REPO), "--literal-pathspecs", *args], capture_output=True
    )


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
    tree, _, rest = proc.stdout.decode().partition("\0")
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
    if not previous_base or not all(map(has_commit, (previous, previous_base, base))):
        return NOTE_UNAVAILABLE.format(sha=previous)
    replayed = replay(previous, previous_base, base)
    if replayed is None:
        return NOTE_UNAVAILABLE.format(sha=previous)
    tree, conflicted = replayed
    clean = [f for f in files if f not in conflicted]
    both = [f for f in files if f in conflicted]
    text = HEADER.format(sha=previous) + diff(tree, clean)
    if both:
        text += CONFLICTED + diff(previous, both)
    return text


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
    files = [f for f in args.files.read_bytes().decode().split("\0") if f]
    try:
        text = interdiff(args.previous, args.previous_base, args.base, files)
    except RuntimeError as e:
        print(f"::warning::interdiff not built: {e}")
        text = NOTE_UNAVAILABLE.format(sha=args.previous)
    if len(text.encode()) > args.max_bytes:
        text = NOTE_TOO_LARGE
    args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
