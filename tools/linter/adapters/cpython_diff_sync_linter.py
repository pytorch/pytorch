"""Lintrunner adapter: keep test/cpython/v3_13 *.py/*.diff/manifest in sync.

Runs the offline verify from test/cpython/diff_sync.py whenever any file under
test/cpython/v3_13/ is in the lint path set. Not a Dynamo test; repo hygiene.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_diff_sync():
    path = REPO_ROOT / "test" / "cpython" / "diff_sync.py"
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
        help="paths to lint (ignored beyond triggering a full offline verify)",
    )
    parser.parse_args()

    diff_sync = _load_diff_sync()
    # Any touch under the include_patterns triggers a full directory verify.
    for err in diff_sync.verify_all():
        rel = err.split(":", 1)[0]
        candidate = REPO_ROOT / "test" / "cpython" / "v3_13" / rel
        path = str(
            candidate
            if candidate.is_file()
            else REPO_ROOT / "test" / "cpython" / "v3_13" / "upstream_manifest.json"
        )
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
                f"{err}\n"
                "Regenerate with: python tools/regenerate_cpython_diffs.py"
            ),
        )
        print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
