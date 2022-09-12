"""
EXEC: Ensure that source files are not executable.
"""
import argparse
import json
import logging
import os
import sys

from enum import Enum
from typing import NamedTuple, Optional

LINTER_CODE = "EXEC"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]


def check_file(filename: str) -> Optional[LintMessage]:
    is_executable = os.access(filename, os.X_OK)
    if is_executable:
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="executable-permissions",
            original=None,
            replacement=None,
            description="This file has executable permission; please remove it by using `chmod -x`.",
        )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exec linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
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

    lint_messages = []
    for filename in args.filenames:
        lint_message = check_file(filename)
        if lint_message is not None:
            lint_messages.append(lint_message)

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
