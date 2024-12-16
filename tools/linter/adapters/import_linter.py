"""
Checks files to make sure there are no imports from disallowed third party
libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from enum import Enum
from typing import List, NamedTuple, Set


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


LINTER_CODE = "NEWLINE"
CURRENT_FILE_NAME = os.path.basename(__file__)
_MODULE_NAME_ALLOW_LIST: Set[str] = set()

# Add builtin modules.
if sys.version_info >= (3, 10):
    _MODULE_NAME_ALLOW_LIST.update(sys.stdlib_module_names)
else:
    from stdlib_list import stdlib_list

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    for name in stdlib_list(version):
        # Skip things like 'ctypes.macholib'
        if "." not in name:
            _MODULE_NAME_ALLOW_LIST.add(name)

# Add the allowed third party libraries.
_MODULE_NAME_ALLOW_LIST.update(
    [
        "torch",
        "sympy",
        "torch_xla",
        "_pytest",
        "functorch",
        "the",
        "libfb",
        "typing_extensions",
        "triton",
        "numpy",
        "torchrec",
        "tabulate",
        "optree",
        "tvm",
    ]
)


def check_file(filename: str) -> List[LintMessage]:
    with open(filename) as f:
        lines = f.readlines()

    # The pattern: from/import word_that_doesn't_start_with_dot
    pattern = re.compile(r"^(?:import|from)\s+([a-zA-Z_][\w]*)")

    lint_messages = []
    for line_number, line in enumerate(lines):
        line_number += 1
        line = line.lstrip()
        match = pattern.search(line)
        if match:
            module_name = match.group(1)
            if module_name not in _MODULE_NAME_ALLOW_LIST:
                msg = LintMessage(
                    path=filename,
                    line=line_number,
                    char=None,
                    code="IMPORT",
                    severity=LintSeverity.ERROR,
                    name="Disallowed import",
                    original=None,
                    replacement=None,
                    description=f"""
importing from {module_name} is not allowed, if you believe there's a valid
reason, please add it to _MODULE_NAME_ALLOW_LIST in {CURRENT_FILE_NAME}
""",
                )
                lint_messages.append(msg)
    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    # Check all files.
    all_lint_messages = []
    for filename in args.filenames:
        lint_messages = check_file(filename)
        all_lint_messages.extend(lint_messages)

    # Print out lint messages.
    for lint_message in all_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
