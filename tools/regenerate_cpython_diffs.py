#!/usr/bin/env python3
"""Fetch pristine CPython sources, regenerate test/cpython/v3_13/*.diff, update manifest.

Usage:
  python tools/regenerate_cpython_diffs.py              # all pairs
  python tools/regenerate_cpython_diffs.py --only test_bool.py
  python tools/regenerate_cpython_diffs.py --check       # verify only (no network)
  python tools/regenerate_cpython_diffs.py --write-header-assertions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO_ROOT / "test"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from cpython.diff_sync import (  # noqa: E402
    CPYTHON_DIR,
    apply_diff_to_adapted,
    fetch_pristine,
    iter_diff_pairs,
    make_unified_diff,
    normalize_bytes,
    parse_header,
    save_manifest,
    sha256_bytes,
    verify_all,
)


ASSERTIONS_HEADER_SNIPPET = """\
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_unittest/test_assertions.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch.testing._internal.common_utils import run_tests


__TestCase = torch._dynamo.test_case.CPythonTestCase
"""


def fix_assertions_header() -> None:
    path = CPYTHON_DIR / "test_unittest" / "test_assertions.py"
    text = path.read_text(encoding="utf-8")
    if "raw.githubusercontent.com/python/cpython" in text:
        print("test_assertions.py already has source URL")
        return
    # Replace the existing Dynamo patch header block through __TestCase assignment.
    end_marker = "__TestCase = torch._dynamo.test_case.CPythonTestCase"
    end = text.find(end_marker)
    if end < 0:
        raise RuntimeError("could not find __TestCase assignment in test_assertions.py")
    end = end + len(end_marker)
    # Keep a single trailing newline before the rest (redirect imports, etc.)
    rest = text[end:].lstrip("\n")
    path.write_text(ASSERTIONS_HEADER_SNIPPET + "\n\n" + rest, encoding="utf-8")
    print(f"updated header: {path.relative_to(REPO_ROOT)}")


def regenerate(only: str | None, *, force: bool = False) -> int:
    """Update manifest for all pairs; rewrite .diff only when stale (or --force)."""
    from cpython.diff_sync import load_manifest

    manifest = load_manifest()
    manifest.setdefault("files", {})
    pairs = iter_diff_pairs()
    if only:
        pairs = [
            (py, diff)
            for py, diff in pairs
            if only in py.relative_to(CPYTHON_DIR).as_posix() or only in py.name
        ]
        if not pairs:
            print(f"no pairs matched --only {only!r}", file=sys.stderr)
            return 1

    failures = 0
    rewritten = 0
    kept = 0
    for py_path, diff_path in pairs:
        rel = py_path.relative_to(CPYTHON_DIR).as_posix()
        repo_rel = py_path.relative_to(REPO_ROOT).as_posix()
        try:
            tag, upstream = parse_header(py_path)
            pristine = fetch_pristine(tag, upstream)
            adapted = normalize_bytes(py_path.read_bytes())
            digest = sha256_bytes(pristine)

            needs_rewrite = force
            if not needs_rewrite:
                try:
                    applied = apply_diff_to_adapted(pristine, py_path, diff_path)
                    needs_rewrite = applied != adapted
                except Exception:
                    needs_rewrite = True

            if needs_rewrite:
                new_diff = make_unified_diff(pristine, adapted, repo_rel)
                diff_path.write_text(new_diff, encoding="utf-8", newline="\n")
                applied = apply_diff_to_adapted(pristine, py_path, diff_path)
                if applied != adapted:
                    raise RuntimeError(
                        "regenerated diff does not reproduce adapted file"
                    )
                rewritten += 1
                print(f"REWRITE  {rel}  ({tag})")
            else:
                kept += 1
                print(f"KEEP     {rel}  ({tag})")

            manifest["files"][rel] = {
                "tag": tag,
                "upstream": upstream,
                "sha256": digest,
            }
        except Exception as e:
            failures += 1
            print(f"FAIL {rel}: {e}", file=sys.stderr)

    save_manifest(manifest)
    print(
        f"done: kept={kept} rewritten={rewritten} failures={failures} "
        f"manifest_entries={len(manifest['files'])}"
    )
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Substring filter on relative path")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rewrite every .diff even if apply(pristine, diff) already matches",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify sync offline using upstream_manifest.json (no network)",
    )
    ap.add_argument(
        "--write-header-assertions",
        action="store_true",
        help="Add missing source URL to test_unittest/test_assertions.py",
    )
    args = ap.parse_args()

    if args.write_header_assertions:
        fix_assertions_header()
        return 0
    if args.check:
        errors = verify_all()
        if errors:
            print("cpython diff sync check FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(f"OK: {len(list(iter_diff_pairs()))} cpython .py/.diff pairs in sync")
        return 0
    return regenerate(args.only, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
