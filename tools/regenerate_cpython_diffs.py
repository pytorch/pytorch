#!/usr/bin/env python3
"""Reconstruct pristine CPython sources and regenerate test/cpython/v3_13/*.diff.

Diffs are written with `git diff --full-index` so the `index <before>..<after>`
line carries blob hashes used by the offline sync check (no manifest file).

This tool never downloads over the network. It reconstructs pristine bytes by
reverse-applying the checked-in .diff. If that fails, or if reverse-apply
would move the checked-in index <before> hash (an edit outside existing
hunks), pass a locally downloaded file via --pristine (see the error hint
for curl/wget). --force still refuses to move <before> without --pristine.

Usage:
  python tools/regenerate_cpython_diffs.py              # all pairs
  python tools/regenerate_cpython_diffs.py --only test_bool.py
  python tools/regenerate_cpython_diffs.py --check       # verify only
  python tools/regenerate_cpython_diffs.py --force       # rewrite every .diff
  python tools/regenerate_cpython_diffs.py --force --pristine /tmp/pristine.py --only test_bool.py
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


def _resolve_pristine(
    py_path: Path,
    diff_path: Path,
    tag: str,
    upstream: str,
    pristine_path: Path | None,
) -> bytes:
    """Prefer offline reverse-apply; optional local --pristine file (no network)."""
    if pristine_path is not None:
        return diff_sync.normalize_bytes(pristine_path.read_bytes())
    try:
        return diff_sync.reverse_apply_to_pristine(py_path, diff_path)
    except Exception as e:
        raise RuntimeError(diff_sync.pristine_download_hint(tag, upstream)) from e


def _filter_pairs(
    pairs: list[tuple[Path, Path]], only: str, cpython_dir: Path
) -> list[tuple[Path, Path]]:
    """Exact relative path, basename, or unique stem (no substring surprises)."""
    exact = [
        (py, diff)
        for py, diff in pairs
        if only == py.relative_to(cpython_dir).as_posix() or only == py.name
    ]
    if exact:
        return exact
    # --only test_bool (no .py): must be unique (test_int != test_int_literal).
    stem = [(py, diff) for py, diff in pairs if only == py.stem]
    if len(stem) == 1:
        return stem
    if len(stem) > 1:
        names = ", ".join(py.name for py, _ in stem)
        raise ValueError(
            f"--only {only!r} matches multiple files ({names}); "
            f"pass an exact basename like {stem[0][0].name!r}"
        )
    return []


def regenerate(
    only: str | None,
    *,
    force: bool = False,
    pristine_path: Path | None = None,
) -> int:
    """Rewrite .diff files when stale (or --force). Writes only if all pairs OK."""
    CPYTHON_DIR = diff_sync.CPYTHON_DIR
    diff_paths = diff_sync.iter_diff_paths()
    pairs = [
        (p.with_suffix(".py"), p) for p in diff_paths if p.with_suffix(".py").is_file()
    ]
    if only:
        try:
            pairs = _filter_pairs(pairs, only, CPYTHON_DIR)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        if not pairs:
            print(f"no pairs matched --only {only!r}", file=sys.stderr)
            return 1
        matched = ", ".join(py.relative_to(CPYTHON_DIR).as_posix() for py, _ in pairs)
        print(f"--only {only!r} matched: {matched}")

    if pristine_path is not None and len(pairs) != 1:
        print(
            "--pristine requires exactly one pair; pass --only <test_name.py>",
            file=sys.stderr,
        )
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
            pristine = _resolve_pristine(
                py_path, diff_path, tag, upstream, pristine_path
            )
            adapted = diff_sync.normalize_bytes(py_path.read_bytes())

            needs_rewrite = force
            if not needs_rewrite:
                errors = diff_sync.verify_pair(py_path, diff_path)
                needs_rewrite = bool(errors)

            if needs_rewrite:
                if pristine_path is None:
                    diff_sync.check_pristine_anchor(
                        pristine,
                        diff_path.read_text(encoding="utf-8", errors="strict"),
                        tag,
                        upstream,
                    )
                new_diff = diff_sync.make_unified_diff(pristine, adapted, repo_rel)
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_diff = Path(tmp) / "patch.diff"
                    diff_sync.write_utf8(tmp_diff, new_diff)
                    applied = diff_sync.apply_diff_to_adapted(
                        pristine, py_path, tmp_diff
                    )
                    if applied != adapted:
                        raise RuntimeError(
                            "regenerated diff does not reproduce adapted file"
                        )
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
        diff_sync.write_utf8(diff_path, new_diff)

    print(f"done: kept={kept} rewritten={rewritten} failures=0")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default=None,
        help="Exact basename, relative path, or unique stem (e.g. test_bool.py)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rewrite every .diff even if verify already passes "
            "(still refuses to move index <before> without --pristine)"
        ),
    )
    ap.add_argument(
        "--pristine",
        type=Path,
        default=None,
        help="Local pristine upstream file (no network fetch); requires --only",
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
            for rel, err in errors:
                print(f"  - {rel}: {err}")
            return 1
        n = len(list(diff_sync.iter_diff_paths()))
        print(f"OK: {n} cpython .py/.diff pairs in sync")
        return 0
    return regenerate(args.only, force=args.force, pristine_path=args.pristine)


if __name__ == "__main__":
    sys.exit(main())
