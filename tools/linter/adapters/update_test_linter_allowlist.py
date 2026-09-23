#!/usr/bin/env python3
"""Regenerate the TEST_LINTER allowlist from actual linter results.

Walks every file matched by [linter.TEST_LINTER] in .lintrunner.toml, runs the
linter's checks on it (ignoring the current allowlist), and writes the sorted
repo-relative paths of the failing files back to test_linter_allowlist.json.

    python tools/linter/adapters/update_test_linter_allowlist.py          # write
    python tools/linter/adapters/update_test_linter_allowlist.py --check  # report drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parents[2]
ALLOWLIST_PATH = ADAPTER_DIR / "test_linter_allowlist.json"

# Running this file directly only puts its own directory on sys.path, so add the
# repo root and import the linter as a package module (which is also how type
# checkers resolve it).
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.adapters import test_linter as tl


# Mirrors the include_patterns / exclude_patterns of [linter.TEST_LINTER].
INCLUDE_PATTERNS = ("test/**/test_*.py", "test/**/*_test.py")
EXCLUDE_PREFIXES = (
    "test/cpp_extensions/open_registration_extension/",
    "test/cpython/",
)

# A file that does not parse is not "not yet migrated": allowlisting it would
# hide a syntax error, so report it and keep it failing in CI instead.
NON_MIGRATION_ERRORS = {"[parse_error]"}


def discover_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in INCLUDE_PATTERNS:
        for path in REPO_ROOT.glob(pattern):
            if not tl._is_test_file(str(path)):
                continue
            rel_path = path.relative_to(REPO_ROOT).as_posix()
            if rel_path.startswith(EXCLUDE_PREFIXES):
                continue
            files.add(path)
    return sorted(files)


def collect_failing_files(files: list[Path]) -> tuple[list[str], list[Path]]:
    # check_file() returns nothing for files already in the allowlist, which
    # would drop every existing entry; clear it so the list is rebuilt from the
    # linter's real output rather than from its current contents.
    tl._allowlist = set()

    entries: list[str] = []
    unparsable: list[Path] = []
    for path in files:
        messages = tl.check_file(str(path))
        if any(m.name in NON_MIGRATION_ERRORS for m in messages):
            unparsable.append(path)
        elif messages:
            entries.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(entries), unparsable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing the allowlist",
    )
    args = parser.parse_args()

    files = discover_files()
    entries, unparsable = collect_failing_files(files)

    for path in unparsable:
        print(
            f"warning: {path.relative_to(REPO_ROOT)}: "
            "failed to parse Python source; excluded from the generated allowlist",
            file=sys.stderr,
        )

    old_content = (
        ALLOWLIST_PATH.read_text(encoding="utf-8") if ALLOWLIST_PATH.exists() else ""
    )
    old: list[str] = json.loads(old_content) if old_content else []
    added = sorted(set(entries) - set(old))
    removed = sorted(set(old) - set(entries))
    print(f"Checked {len(files)} test files; {len(entries)} require allowlisting.")
    print(f"Allowlist changes: {len(added)} added, {len(removed)} removed.")
    for path in added:
        print(f"  + {path}")
    for path in removed:
        print(f"  - {path}")

    new_content = json.dumps(entries, indent=2) + "\n"
    if args.check:
        if new_content != old_content:
            print(f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} is out of date")
            sys.exit(1)
        print(f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} is up to date")
        return

    ALLOWLIST_PATH.write_text(new_content, encoding="utf-8")
    print(f"wrote {len(entries)} entries to {ALLOWLIST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
