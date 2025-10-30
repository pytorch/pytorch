from __future__ import annotations

import argparse
import json
import logging
import os
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


def run_command(args: list[str]) -> subprocess.CompletedProcess[bytes]:
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


def check_run_typos(filenames: list[str], config: str, code: str) -> list[LintMessage]:
    filenames = [os.path.relpath(f) for f in filenames]
    try:
        proc = run_command(
            ["typos", "--config", f"{config}", "--format", "json"] + filenames,
        )
        # print(["typos", "--config", f"{config}"] + filenames)
    except OSError:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="Typos",
                original=None,
                replacement=None,
                description=None,
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    rc = []
    for line in stdout.splitlines():
        if not line:
            continue
        dic = json.loads(line)
        rc.append(
            LintMessage(
                path=dic.get("path", None),
                line=dic.get("line_num", None),
                char=dic.get("byte_offset", None),
                code=code,
                severity=LintSeverity.ERROR,
                name="Typos",
                original=dic.get("typo", None),
                replacement=dic.get("corrections", None)[0]
                if dic.get("corrections", None)
                else None,
                description=f"{dic.get('typo', None)} should be replaced by {dic.get('corrections', None)}",
            )
        )
    return rc


def check_typos_installed(code: str) -> list[LintMessage]:
    cmd = ["typos", "-V"]
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
                name="Typos",
                original=None,
                replacement=None,
                description=msg,
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="typos wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to an typos _typos.toml file",
    )
    parser.add_argument(
        "--code",
        default="TYPOS",
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
    # print(args.filenames)
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    lint_messages = check_typos_installed(args.code) + check_run_typos(
        args.filenames, args.config, args.code
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
