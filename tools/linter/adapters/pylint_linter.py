from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
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
            check=False,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def check_pylint_installed(code: str) -> list[LintMessage]:
    cmd = [sys.executable, "-mpylint", "--version"]
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
    filenames: list[str],
    config: str,
    code: str,
) -> list[LintMessage]:
    try:
        proc = run_command(
            ["pylint", f"--rcfile={config}", "-f", "json"] + filenames,
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
    if proc.returncode == 32:
        stderr = str(proc.stderr, "utf-8").strip()
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
    stdout = str(proc.stdout, "utf-8").strip()
    errors = json.loads(stdout)

    return [
        LintMessage(
            path=error["path"],
            name=error["message-id"],
            description=error["message"],
            line=int(error["line"]),
            char=int(error["column"]),
            code=code,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for error in errors
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pylint wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to a pylintrc config file",
    )
    parser.add_argument(
        "--code",
        default="PYLINT",
        help="the code this lint should report as",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    filenames: set[str] = set()

    # If a stub file exists, have pylint check it instead of the original file, in
    # accordance with PEP-484 (see https://www.python.org/dev/peps/pep-0484/#stub-files)
    for filename in args.filenames:
        if filename.endswith(".pyi"):
            filenames.add(filename)
            continue

        stub_filename = filename.replace(".py", ".pyi")
        if Path(stub_filename).exists():
            filenames.add(stub_filename)
        else:
            filenames.add(filename)

    lint_messages = check_pylint_installed(args.code) + check_files(
        list(filenames), args.config, args.code
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
