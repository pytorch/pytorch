#!/usr/bin/env python3
"""Fetch/reconstruct pristine CPython sources and regenerate test/cpython/v3_13/*.diff.

Diffs are written with `git diff --full-index` so the `index <before>..<after>`
line carries blob hashes used by the offline sync check (no manifest file).

Usage:
  python tools/regenerate_cpython_diffs.py              # all pairs
  python tools/regenerate_cpython_diffs.py --only test_bool.py
  python tools/regenerate_cpython_diffs.py --check       # verify only (no network)
  python tools/regenerate_cpython_diffs.py --force       # rewrite every .diff
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_diff_sync() -> ModuleType:
    import importlib.util

    path = REPO_ROOT / "tools" / "cpython_diff_sync.py"
    spec = importlib.util.spec_from_file_location("torch_cpython_diff_sync", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diff_sync = _load_diff_sync()


def _resolve_pristine(py_path: Path, diff_path: Path, tag: str, upstream: str) -> bytes:
    """Prefer offline reverse-apply; fall back to network fetch."""
    try:
        return diff_sync.reverse_apply_to_pristine(py_path, diff_path)
    except Exception:
        return diff_sync.fetch_pristine(tag, upstream)


def regenerate(only: str | None, *, force: bool = False) -> int:
    """Rewrite .diff files when stale (or --force). Writes only if all pairs OK."""
    CPYTHON_DIR = diff_sync.CPYTHON_DIR
    diff_paths = diff_sync.iter_diff_paths()
    pairs = [(p.with_suffix(".py"), p) for p in diff_paths if p.with_suffix(".py").is_file()]
    if only:
        pairs = [
            (py, diff)
            for py, diff in pairs
            if only in py.relative_to(CPYTHON_DIR).as_posix() or only in py.name
        ]
        if not pairs:
            print(f"no pairs matched --only {only!r}", file=sys.stderr)
            return 1

    planned: list[tuple[Path, str]] = []
    failures = 0
    rewritten = 0
    kept = 0

    for py_path, diff_path in pairs:
        rel = py_path.relative_to(CPYTHON_DIR).as_posix()
        repo_rel = py_path.relative_to(REPO_ROOT).as_posix()
        try:
            tag, upstream = diff_sync.parse_header(py_path)
            pristine = _resolve_pristine(py_path, diff_path, tag, upstream)
            adapted = diff_sync.normalize_bytes(py_path.read_bytes())

            needs_rewrite = force
            if not needs_rewrite:
                errors = diff_sync.verify_pair(py_path, diff_path)
                needs_rewrite = bool(errors)

            if needs_rewrite:
                new_diff = diff_sync.make_unified_diff(pristine, adapted, repo_rel)
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_diff = Path(tmp) / "patch.diff"
                    tmp_diff.write_text(new_diff, encoding="utf-8", newline="\n")
                    applied = diff_sync.apply_diff_to_adapted(
                        pristine, py_path, tmp_diff
                    )
                    if applied != adapted:
                        raise RuntimeError(
                            "regenerated diff does not reproduce adapted file"
                        )
                    # Confirm full-index hashes round-trip.
                    before, after = diff_sync.parse_index(new_diff)
                    if diff_sync.git_hash_object(pristine) != before:
                        raise RuntimeError("before hash mismatch in regenerated diff")
                    if diff_sync.git_hash_object(adapted) != after:
                        raise RuntimeError("after hash mismatch in regenerated diff")
                planned.append((diff_path, new_diff))
                rewritten += 1
                print(f"REWRITE  {rel}  ({tag})")
            else:
                kept += 1
                print(f"KEEP     {rel}  ({tag})")
        except Exception as e:
            failures += 1
            print(f"FAIL {rel}: {e}", file=sys.stderr)

    if failures:
        print(
            f"NOT writing diffs ({failures} failure(s)); "
            f"kept={kept} rewritten={rewritten}",
            file=sys.stderr,
        )
        return 1

    for diff_path, new_diff in planned:
        diff_path.write_text(new_diff, encoding="utf-8", newline="\n")

    print(f"done: kept={kept} rewritten={rewritten} failures=0")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Substring filter on relative path")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rewrite every .diff even if verify already passes",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify sync offline via full-index lines (no network)",
    )
    args = ap.parse_args()

    if args.check:
        errors = diff_sync.verify_all()
        if errors:
            print("cpython diff sync check FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1
        n = len(list(diff_sync.iter_diff_paths()))
        print(f"OK: {n} cpython .py/.diff pairs in sync")
        return 0
    return regenerate(args.only, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
