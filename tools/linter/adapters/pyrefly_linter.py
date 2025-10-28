from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from typing import NamedTuple


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


# Note: This regex pattern is kept for reference but not used for pyrefly JSON parsing
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<severity>\S+?):?
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)

# torch/_dynamo/variables/tensor.py:363: error: INTERNAL ERROR
INTERNAL_ERROR_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    \s(?P<severity>\S+?):?
    \s(?P<message>INTERNAL\sERROR.*)
    $
    """
)


def run_command(
    args: list[str],
    *,
    extra_env: dict[str, str] | None,
    retries: int,
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


# Severity mapping (currently only used for stderr internal errors)
# Pyrefly JSON output doesn't include severity, so all errors default to ERROR
severities = {
    "error": LintSeverity.ERROR,
    "note": LintSeverity.ADVICE,
}


def check_pyrefly_installed(code: str) -> list[LintMessage]:
    cmd = ["pyrefly", "--version"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return []
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace")
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=f"Could not run '{' '.join(cmd)}': {msg}",
            )
        ]


def in_github_actions() -> bool:
    return bool(os.getenv("GITHUB_ACTIONS"))


def check_files(
    code: str,
    config: str,
) -> list[LintMessage]:
    try:
        pyrefly_commands = [
            "pyrefly",
            "check",
            "--config",
            config,
            "--output-format=json",
        ]
        proc = run_command(
            [*pyrefly_commands],
            extra_env={},
            retries=0,
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    stderr = str(proc.stderr, "utf-8").strip()
    if proc.returncode not in (0, 1):
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=stderr,
            )
        ]

    # Parse JSON output from pyrefly
    try:
        if stdout:
            result = json.loads(stdout)
            errors = result.get("errors", [])
        else:
            errors = []
        # For now filter out deprecated warnings and only report type errors as warnings
        # until we remove mypy
        errors = [error for error in errors if error["name"] != "deprecated"]
        rc = [
            LintMessage(
                path=error["path"],
                name=error["name"],
                description=error.get(
                    "description", error.get("concise_description", "")
                ),
                line=error["line"],
                char=error["column"],
                code=code,
                severity=LintSeverity.ADVICE,
                # uncomment and replace when we switch to pyrefly
                # severity=LintSeverity.ADVICE if error["name"] == "deprecated" else LintSeverity.ERROR,
                original=None,
                replacement=None,
            )
            for error in errors
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="json-parse-error",
                original=None,
                replacement=None,
                description=f"Failed to parse pyrefly JSON output: {e}",
            )
        ]

    # Still check stderr for internal errors
    rc += [
        LintMessage(
            path=match["file"],
            name="INTERNAL ERROR",
            description=match["message"],
            line=int(match["line"]),
            char=None,
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in INTERNAL_ERROR_RE.finditer(stderr)
    ]
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pyrefly wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--code",
        default="PYREFLY",
        help="the code this lint should report as",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to an mypy .ini config file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    lint_messages = check_pyrefly_installed(args.code) + check_files(
        args.code, args.config
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
