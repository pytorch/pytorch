"""
CONSTEXPR: Ensures users don't use vanilla constexpr since it causes issues
"""

import argparse
import json
import logging
import sys

from enum import Enum
from typing import NamedTuple, Optional

CONSTEXPR = "constexpr char"
CONSTEXPR_MACRO = "CONSTEXPR_EXCEPT_WIN_CUDA char"

LINTER_CODE = "CONSTEXPR"


class LintSeverity(str, Enum):
    ERROR = "error"


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
    logging.debug("Checking file %s", filename)

    with open(filename) as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if CONSTEXPR in line:
            original = "".join(lines)
            replacement = original.replace(CONSTEXPR, CONSTEXPR_MACRO)
            logging.debug("replacement: %s", replacement)
            return LintMessage(
                path=filename,
                line=idx,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Vanilla constexpr used, prefer macros",
                original=original,
                replacement=replacement,
                description="Vanilla constexpr used, prefer macros run `lintrunner --take CONSTEXPR -a` to apply changes.",
            )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CONSTEXPR linter",
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
