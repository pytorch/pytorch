#!/usr/bin/env python3
"""
Checks that all symbols in torch/header_only_apis.txt are tested in a .cpp
test file to ensure header-only-ness. The .cpp test file must be built
without linking libtorch.
"""

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "HEADER_ONLY_LINTER"


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


CPP_TEST_GLOBS = [
    "test/cpp/aoti_abi_check/*.cpp",
]

REPO_ROOT = Path(__file__).parents[3]


def found_symbol(symbol: str) -> bool:
    # check that the symbol is in a noncommented out region of the test files
    for cpp_test_glob in CPP_TEST_GLOBS:
        for test_file in REPO_ROOT.glob(cpp_test_glob):
            with open(test_file) as tf:
                for test_file_line in tf:
                    if not test_file_line.startswith("//") and symbol in test_file_line:
                        return True
    return False


def check_file(filename: str) -> list[LintMessage]:
    lint_messages = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            # commented out lines should be skipped
            if line.startswith("#"):
                continue

            symbol = line.strip()
            if symbol == "":
                continue

            if not found_symbol(symbol):
                lint_messages.append(
                    LintMessage(
                        path=filename,
                        line=idx + 1,
                        char=None,
                        code=LINTER_CODE,
                        severity=LintSeverity.ERROR,
                        name="[untested-symbol]",
                        original=None,
                        replacement=None,
                        description=(
                            f"{symbol} has been included as a header-only API "
                            "but is not tested in any of CPP_TEST_GLOBS. "
                            "Please add a .cpp test using the symbol without "
                            "linking anything to verify that the symbol is in "
                            "fact header-only. If not already there, please "
                            "add the .cpp file to CPP_TEST_GLOBS in "
                            "tools/linters/adapters/header_only_linter.py."
                        ),
                    )
                )

    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="header only APIs linter",
        fromfile_prefix_chars="@",
    )
    args = parser.parse_args()

    for lint_message in check_file(str(REPO_ROOT) + "/torch/header_only_apis.txt"):
        print(json.dumps(lint_message._asdict()), flush=True)
