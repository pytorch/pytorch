"""
NEWLINE: Checks files to make sure there are no trailing newlines.
"""
import argparse
import json
import logging
import os
import sys

from enum import Enum
from typing import NamedTuple, Optional

NEWLINE = 10  # ASCII "\n"
LINTER_CODE = "NEWLINE"


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


def correct_trailing_newlines(filename: str) -> bool:
    with open(filename, "rb") as f:
        a = len(f.read(2))
        if a == 0:
            return True
        elif a == 1:
            # file is wrong whether or not the only byte is a newline
            return False
        else:
            f.seek(-2, os.SEEK_END)
            b, c = f.read(2)
            # no ASCII byte is part of any non-ASCII character in UTF-8
            return b != NEWLINE and c == NEWLINE


def check_file(filename: str) -> Optional[LintMessage]:
    logging.debug("Checking file %s", filename)

    with open(filename, "rb") as f:
        a = len(f.read(2))
        if a == 0:
            # File is empty, just leave it alone.
            return None
        elif a == 1:
            # file is wrong whether or not the only byte is a newline
            return LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="testestTrailing newline",
                original=None,
                replacement=None,
                description="Trailing newline found. Run `lintunner --take NEWLINE -a` to apply changes.",
            )

        else:
            # Read the last two bytes
            f.seek(-2, os.SEEK_END)
            b, c = f.read(2)
            # no ASCII byte is part of any non-ASCII character in UTF-8
            if b != NEWLINE and c == NEWLINE:
                return None
            else:
                f.seek(0)
                try:
                    original = f.read().decode("utf-8")
                except Exception as err:
                    return LintMessage(
                        path=filename,
                        line=None,
                        char=None,
                        code=LINTER_CODE,
                        severity=LintSeverity.ERROR,
                        name="Decoding failure",
                        original=None,
                        replacement=None,
                        description=f"utf-8 decoding failed due to {err.__class__.__name__}:\n{err}",
                    )

                return LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="Trailing newline",
                    original=original,
                    replacement=original.rstrip("\n") + "\n",
                    description="Trailing newline found. Run `lintunner --take NEWLINE -a` to apply changes.",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="location of native_functions.yaml",
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
