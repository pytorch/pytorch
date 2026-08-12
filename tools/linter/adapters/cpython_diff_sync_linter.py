"""Lintrunner adapter: keep test/cpython/v3_13 *.py/*.diff pairs in sync.

Runs the offline verify from tools/cpython_diff_sync.py whenever any file under
test/cpython/v3_13/ is in the lint path set. Not a Dynamo test; repo hygiene.

verify_all() always checks the whole tree. include_patterns are only a trigger
so the linter runs; do not "optimize" the adapter to per-file verify or drift
under untouched pairs will silently slip through.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_diff_sync():
    path = REPO_ROOT / "tools" / "cpython_diff_sync.py"
    spec = importlib.util.spec_from_file_location("torch_cpython_diff_sync", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify test/cpython/v3_13 adapted tests stay in sync with *.diff",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="paths to lint (trigger only; verify_all always checks the whole tree)",
    )
    parser.parse_args()

    diff_sync = _load_diff_sync()
    for rel, err in diff_sync.verify_all():
        candidate = REPO_ROOT / "test" / "cpython" / "v3_13" / rel
        if not candidate.is_file() and not rel.endswith(".diff"):
            candidate = REPO_ROOT / "test" / "cpython" / "v3_13" / (rel + ".diff")
        path = str(candidate if candidate.exists() else diff_sync.CPYTHON_DIR)
        msg = LintMessage(
            path=path,
            line=1,
            char=None,
            code="CPYTHON_DIFF_SYNC",
            severity=LintSeverity.ERROR,
            name="cpython-diff-out-of-sync",
            original=None,
            replacement=None,
            description=(
                f"{rel}: {err}\n"
                "Regenerate with: python tools/regenerate_cpython_diffs.py"
            ),
        )
        print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
