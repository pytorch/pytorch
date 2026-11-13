# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "packaging==25.0",
#   "tomli==2.2.1 ; python_version < '3.11'",
# ]
# ///
"""Lintrunner adapter: PEP 639 license-files / SPDX alignment (issue #183434)."""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.license_files_audit import audit_repo_license_files


sys.path.remove(str(REPO_ROOT))

LINTER_CODE = "LICENSE_FILES"


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


def _emit(
    report_path: str, severity: LintSeverity, name: str, description: str
) -> None:
    lint = LintMessage(
        path=report_path,
        line=None,
        char=None,
        code=LINTER_CODE,
        severity=severity,
        name=name,
        original=None,
        replacement=None,
        description=description,
    )
    payload = lint._asdict()
    payload["severity"] = lint.severity.value
    print(json.dumps(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pyproject.toml license-files vs source tree.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths that triggered this linter (audit always uses full repo)",
    )
    parser.parse_args()  # filenames from lintrunner; audit always scans full REPO_ROOT
    errors, skip_reason = audit_repo_license_files(REPO_ROOT)
    if skip_reason:
        # Any emitted message, ADVICE included, fails lintrunner. A checkout
        # without every submodule is normal locally; quick-checks is the run
        # that must not skip (see license_files_audit.main).
        return
    for msg in errors:
        _emit("pyproject.toml", LintSeverity.ERROR, "license-files-audit", msg)


if __name__ == "__main__":
    main()
