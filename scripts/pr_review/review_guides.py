#!/usr/bin/env python3
"""List the trusted REVIEW.md guides that cover a pull request's changed paths.

pr-review applies a directory's REVIEW.md to every changed file under that
directory, at any depth. The review job takes those guides from the trusted
checkout of the default branch, never from the pull request: a PR that could
supply its own guides could relax or delete the rules that judge it. Its change
to a guide is reviewed like any other change.

The changed paths are NUL-separated `git diff --no-renames --name-only -z`
output, so both sides of a rename and every deleted path are present: a guide
still covers a file that moves out from under it or is deleted. They are
pull-request data and only select directories. What is written out is the
absolute path of each REVIEW.md that exists in the trusted checkout, root first,
one per line; nothing in it is spelled by the pull request. An empty file means
no guide applies.

Exit 0 on success; 2 on a usage or read error, on a changed path that is not a
normalized relative path (git never emits one), or on an applicable guide whose
path holds a control character. Each fails the step closed rather than
reviewing without a guide that should apply.

Usage: review_guides.py <trusted-root> <changed-paths.z> <out>
"""

from __future__ import annotations

import os
import sys


GUIDE = "REVIEW.md"


def covering_dirs(path: str) -> list[str] | None:
    """Every directory above `path`, the repository root ("") first.

    None when `path` is not a normalized relative path.
    """
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    return ["/".join(parts[:i]) for i in range(len(parts))]


def applicable_guides(root: str, paths: list[str]) -> list[str] | None:
    """Repo-relative paths of the guides in `root` covering `paths`, root first.

    None when a path is malformed, or when a guide that applies has a control
    character in its path and so cannot be written one per line. A guide
    counts only as a regular file whose real path stays inside `root`, so a
    symlink cannot widen what is read.
    """
    real_root = os.path.realpath(root)
    found = set()
    for path in paths:
        dirs = covering_dirs(path)
        if dirs is None:
            return None
        for d in dirs:
            rel = f"{d}/{GUIDE}" if d else GUIDE
            candidate = os.path.join(real_root, rel)
            if os.path.islink(candidate) or not os.path.isfile(candidate):
                continue
            real = os.path.realpath(candidate)
            if os.path.commonpath([real_root, real]) != real_root:
                continue
            if any(ord(c) < 0x20 for c in rel):
                return None
            found.add(rel)
    return sorted(found, key=lambda rel: (rel.count("/"), rel))


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    root, paths_file, out = argv[1:]
    try:
        with open(paths_file, "rb") as f:
            raw = f.read()
    except OSError as exc:
        print(f"cannot read the changed paths: {exc}", file=sys.stderr)
        return 2
    paths = [os.fsdecode(p) for p in raw.split(b"\0") if p]
    guides = applicable_guides(root, paths)
    if guides is None:
        # The path itself is not printed: it is pull-request data, and this
        # goes to a log the runner parses for workflow commands.
        print(
            "a changed path is not a normalized relative path, or a guide that "
            "applies has a control character in its path",
            file=sys.stderr,
        )
        return 2
    base = os.path.abspath(root)
    with open(out, "wb") as f:
        for rel in guides:
            f.write(os.fsencode(os.path.join(base, rel)) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
