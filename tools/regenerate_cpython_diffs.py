#!/usr/bin/env python3
"""Fetch pristine CPython sources, regenerate test/cpython/v3_13/*.diff, update manifest.

Usage:
  python tools/regenerate_cpython_diffs.py              # all pairs
  python tools/regenerate_cpython_diffs.py --only test_bool.py
  python tools/regenerate_cpython_diffs.py --check       # verify only (no network)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_diff_sync() -> ModuleType:
    path = REPO_ROOT / "test" / "cpython" / "diff_sync.py"
    spec = importlib.util.spec_from_file_location("torch_cpython_diff_sync", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diff_sync = _load_diff_sync()


def regenerate(only: str | None, *, force: bool = False) -> int:
    """Update manifest for all pairs; rewrite .diff only when stale (or --force).

    Fetches / computes everything first, then writes .diff files and the manifest
    only if every pair succeeded — avoids leaving a half-updated tree on a
    mid-run network flake.
    """
    CPYTHON_DIR = diff_sync.CPYTHON_DIR
    manifest = diff_sync.load_manifest()
    files: dict = dict(manifest.get("files", {}))
    pairs = diff_sync.iter_diff_pairs()
    if only:
        pairs = [
            (py, diff)
            for py, diff in pairs
            if only in py.relative_to(CPYTHON_DIR).as_posix() or only in py.name
        ]
        if not pairs:
            print(f"no pairs matched --only {only!r}", file=sys.stderr)
            return 1

    planned: list[tuple[Path, str | None, str, dict]] = []
    # (diff_path, new_diff_text_or_None_to_keep, rel, manifest_entry)
    failures = 0
    rewritten = 0
    kept = 0
    seen: set[str] = set()

    for py_path, diff_path in pairs:
        rel = py_path.relative_to(CPYTHON_DIR).as_posix()
        repo_rel = py_path.relative_to(REPO_ROOT).as_posix()
        seen.add(rel)
        try:
            tag, upstream = diff_sync.parse_header(py_path)
            pristine = diff_sync.fetch_pristine(tag, upstream)
            adapted = diff_sync.normalize_bytes(py_path.read_bytes())
            digest = diff_sync.sha256_bytes(pristine)

            needs_rewrite = force
            if not needs_rewrite:
                try:
                    applied = diff_sync.apply_diff_to_adapted(
                        pristine, py_path, diff_path
                    )
                    needs_rewrite = applied != adapted
                except Exception:
                    needs_rewrite = True

            new_diff: str | None = None
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
                rewritten += 1
                print(f"REWRITE  {rel}  ({tag})")
            else:
                kept += 1
                print(f"KEEP     {rel}  ({tag})")

            planned.append(
                (
                    diff_path,
                    new_diff,
                    rel,
                    {"tag": tag, "upstream": upstream, "sha256": digest},
                )
            )
        except Exception as e:
            failures += 1
            print(f"FAIL {rel}: {e}", file=sys.stderr)

    if failures:
        print(
            f"NOT writing diffs/manifest ({failures} failure(s)); "
            f"kept={kept} rewritten={rewritten}",
            file=sys.stderr,
        )
        return 1

    for diff_path, new_diff, rel, entry in planned:
        if new_diff is not None:
            diff_path.write_text(new_diff, encoding="utf-8", newline="\n")
        files[rel] = entry

    if only is None:
        stale = sorted(set(files) - seen)
        for rel in stale:
            del files[rel]
            print(f"PRUNE    {rel}")

    manifest["files"] = files
    diff_sync.save_manifest(manifest)
    print(
        f"done: kept={kept} rewritten={rewritten} failures=0 "
        f"manifest_entries={len(files)}"
    )
    return 0


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
    args = ap.parse_args()

    if args.check:
        errors = diff_sync.verify_all()
        if errors:
            print("cpython diff sync check FAILED:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print(
            f"OK: {len(list(diff_sync.iter_diff_pairs()))} "
            "cpython .py/.diff pairs in sync"
        )
        return 0
    return regenerate(args.only, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
