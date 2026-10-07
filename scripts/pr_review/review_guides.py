#!/usr/bin/env python3
"""Print the REVIEW.md guides that apply to a change, as one markdown document.

Any directory may hold a REVIEW.md with review rules for every file under it, at
any depth. This finds the guides covering each file a change touches, root
first, and prints one section per guide: a heading naming the guide and the
directory it covers, then its text verbatim between begin and end markers. When
no guide applies, it says so.

The guides come from a trusted revision, never from the change: a change that
could supply its own guides could relax or delete the rules that judge it. Its
own REVIEW.md edits are reviewed like any other change.

Usage:

  review_guides.py BASE HEAD
      Run inside a git checkout. Reads the guides from commit BASE and prints
      those covering the files changed between the merge base of BASE and HEAD
      and HEAD. Pass the commit the change is reviewed against: a PR's base
      branch, which for a ghstack PR is its gh/<user>/<n>/base.

  review_guides.py ROOT PATHS OUT
      Reads the guides from the checked-out tree ROOT and writes OUT. PATHS
      holds the changed files as `git diff --no-renames --name-only -z` output.
      The hardened review uses this form: its trusted tree and the PR checkout
      are separate repositories.

Changed files include both sides of a rename and deleted files, since a guide
still covers a file that moves out from under it or is deleted. A guide counts
only as a regular file; a symlink, or anything resolving outside ROOT, is
ignored.

Exit 0 on success; 2 on a usage, git or read error, on a changed path that is
not a normalized relative path (git never emits one), or on an applicable guide
whose path holds a control character. Each fails closed rather than reviewing
without a guide that should apply.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable


GUIDE = "REVIEW.md"

INTRO = (
    "# Directory review guides\n\n"
    "Each section is a REVIEW.md from the trusted base revision that covers files "
    "this change touches. Apply it to the changed files under the directory its "
    "heading names.\n"
)
NO_GUIDES = (
    "# Directory review guides\n\n"
    "No REVIEW.md in the trusted base revision covers the files this change "
    "touches, so no guide applies.\n"
)


def covering_dirs(path: str) -> list[str] | None:
    """Every directory above `path`, the repository root ("") first.

    None when `path` is not a normalized relative path.
    """
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    return ["/".join(parts[:i]) for i in range(len(parts))]


def select(paths: list[str], is_guide: Callable[[str], bool]) -> list[str] | None:
    """Repo-relative paths of the guides covering `paths`, root first.

    None when a path is malformed, or when a guide that applies has a control
    character in its path and so cannot head its section on one line.
    """
    found = set()
    for path in paths:
        dirs = covering_dirs(path)
        if dirs is None:
            return None
        for d in dirs:
            rel = f"{d}/{GUIDE}" if d else GUIDE
            if not is_guide(rel):
                continue
            if any(ord(c) < 0x20 for c in rel):
                return None
            found.add(rel)
    return sorted(found, key=lambda rel: (rel.count("/"), rel))


def render(guides: list[str], read: Callable[[str], bytes]) -> bytes:
    """The combined document for `guides`, each read with `read`."""
    if not guides:
        return NO_GUIDES.encode()
    out = [INTRO.encode()]
    for rel in guides:
        d = rel[: -len(GUIDE)].rstrip("/")
        scope = f"changed files under `{d}/`" if d else "every changed file"
        body = read(rel)
        if body and not body.endswith(b"\n"):
            body += b"\n"
        out += [
            os.fsencode(f"\n## `{rel}`: applies to {scope}\n\n<!-- begin {rel} -->\n"),
            body,
            os.fsencode(f"<!-- end {rel} -->\n"),
        ]
    return b"".join(out)


def applicable_guides(root: str, paths: list[str]) -> list[str] | None:
    """`select` over the checked-out tree `root`."""
    real_root = os.path.realpath(root)

    def is_guide(rel: str) -> bool:
        candidate = os.path.join(real_root, rel)
        if os.path.islink(candidate) or not os.path.isfile(candidate):
            return False
        real = os.path.realpath(candidate)
        return os.path.commonpath([real_root, real]) == real_root

    return select(paths, is_guide)


def combine(root: str, guides: list[str]) -> bytes:
    """`render` reading from the checked-out tree `root`."""
    real_root = os.path.realpath(root)

    def read(rel: str) -> bytes:
        with open(os.path.join(real_root, rel), "rb") as f:
            return f.read()

    return render(guides, read)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def from_git(base: str, head: str) -> bytes | None:
    """The document for the change from merge-base(base, head) to head."""
    merge_base = git("merge-base", base, head).decode().strip()
    raw = git("diff", "--no-renames", "--name-only", "-z", merge_base, head, "--")
    paths = [os.fsdecode(p) for p in raw.split(b"\0") if p]
    # Regular files only (modes 100644 and 100755); a symlink is mode 120000.
    blobs = {}
    for entry in git("ls-tree", "-r", "-z", "--full-tree", base).split(b"\0"):
        meta, _, name = entry.partition(b"\t")
        rel = os.fsdecode(name)
        if rel.split("/")[-1] == GUIDE and meta.split()[:2] in (
            [b"100644", b"blob"],
            [b"100755", b"blob"],
        ):
            blobs[rel] = meta.split()[2].decode()
    guides = select(paths, blobs.__contains__)
    if guides is None:
        return None
    return render(guides, lambda rel: git("cat-file", "blob", blobs[rel]))


FAILED_PATH = (
    "a changed path is not a normalized relative path, or a guide that "
    "applies has a control character in its path"
)


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        try:
            text = from_git(argv[1], argv[2])
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", b"") or b""
            print(f"git failed: {exc}\n{os.fsdecode(detail)}", file=sys.stderr)
            return 2
        if text is None:
            print(FAILED_PATH, file=sys.stderr)
            return 2
        sys.stdout.buffer.write(text)
        return 0
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
        print(FAILED_PATH, file=sys.stderr)
        return 2
    try:
        text = combine(root, guides)
    except OSError as exc:
        print(f"cannot read a guide: {exc}", file=sys.stderr)
        return 2
    with open(out, "wb") as f:
        f.write(text)
    # Paths of files on the default branch, none spelled by the PR.
    print(f"review guides that apply: {len(guides)}")
    for rel in guides:
        print(f"  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
