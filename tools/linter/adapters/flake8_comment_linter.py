"""
FLAKE8_COMMENT: Checks files to make sure # flake8: noqa is only at the top of files
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tokenize
from enum import Enum
from io import StringIO
from typing import NamedTuple


LINTER_CODE = "FLAKE8_COMMENT"


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


def check_file(filename: str) -> LintMessage | None:
    logging.debug("Checking file %s", filename)

    pattern = r"^# flake8:\s*noqa"
    is_start_of_file = True

    with open(filename, encoding="utf-8") as f:
        original = f.read()

        for token in tokenize.generate_tokens(StringIO(original).readline):
            if (
                token.type != tokenize.COMMENT and token.type != tokenize.NL
            ) and is_start_of_file:
                is_start_of_file = False

            if (
                token.type == tokenize.COMMENT
                and not is_start_of_file
                and re.search(pattern, token.string)
            ):
                replacement_lines = original.splitlines()
                replacement_lines[token.start[0] - 1] = ""
                replacement_lines.insert(0, "# flake8: noqa")
                return LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="mid-file '# flake8: noqa'",
                    original=original,
                    replacement="\n".join(replacement_lines),
                    description="'# flake8: noqa' in the middle of the file ",
                )
    return None


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
