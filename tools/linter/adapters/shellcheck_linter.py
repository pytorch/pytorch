# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "shellcheck-py==0.7.2.1; platform_machine == 'x86_64'",
# ]
# ///
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "SHELLCHECK"


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


def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def _is_x86_64() -> bool:
    return platform.machine() == "x86_64"


def _shellcheck_candidates() -> list[str]:
    path_env = os.environ.get("PATH", "")
    candidates: list[str] = []
    for directory in path_env.split(os.pathsep):
        if not directory:
            continue
        candidate = os.path.join(directory, "shellcheck")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            candidates.append(candidate)
    return candidates


def check_files(
    files: list[str],
) -> list[LintMessage]:
    args = ["--external-sources", "--format=json1"] + files

    proc: subprocess.CompletedProcess[bytes] | None = None
    last_error: OSError | None = None

    for shellcheck in _shellcheck_candidates():
        try:
            proc = run_command([shellcheck] + args)
            break
        except OSError as err:
            last_error = err

    if proc is None:
        if last_error is not None and last_error.errno == 8:
            return []
        if not _is_x86_64():
            return []
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed to execute shellcheck.\n{last_error.__class__.__name__}: {last_error}"
                    if last_error is not None
                    else "Failed to find a usable shellcheck executable."
                ),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    results = json.loads(stdout)["comments"]
    return [
        LintMessage(
            path=result["file"],
            name=f"SC{result['code']}",
            description=result["message"],
            line=result["line"],
            char=result["column"],
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for result in results
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="shellcheck runner",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    if not _shellcheck_candidates():
        if not _is_x86_64():
            sys.exit(0)
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="shellcheck is not installed, did you forget to run `lintrunner init`?",
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    args = parser.parse_args()

    lint_messages = check_files(args.filenames)
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
