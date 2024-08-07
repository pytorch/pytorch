#!/usr/bin/env python3
"""
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

This lint verifies that every Python test file (file that matches test_*.py or *_test.py in the test folder)
has valid ownership information in a comment header. Valid means:
  - The format of the header follows the pattern "# Owner(s): ["list", "of owner", "labels"]
  - Each owner label actually exists in PyTorch
  - Each owner label starts with "module: " or "oncall: " or is in ACCEPTABLE_OWNER_LABELS
"""

from __future__ import annotations

import argparse
import json
from enum import Enum
from typing import Any, NamedTuple
from urllib.request import urlopen


LINTER_CODE = "TESTOWNERS"


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


# Team/owner labels usually start with "module: " or "oncall: ", but the following are acceptable exceptions
ACCEPTABLE_OWNER_LABELS = ["NNC", "high priority"]
OWNERS_PREFIX = "# Owner(s): "


def get_pytorch_labels() -> Any:
    labels = (
        urlopen("https://ossci-metrics.s3.amazonaws.com/pytorch_labels.json")
        .read()
        .decode("utf-8")
    )
    return json.loads(labels)


PYTORCH_LABELS = get_pytorch_labels()
# Team/owner labels usually start with "module: " or "oncall: ", but the following are acceptable exceptions
ACCEPTABLE_OWNER_LABELS = ["NNC", "high priority"]
GLOB_EXCEPTIONS = ["**/test/run_test.py"]


def check_labels(
    labels: list[str], filename: str, line_number: int
) -> list[LintMessage]:
    lint_messages = []
    for label in labels:
        if label not in PYTORCH_LABELS:
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=line_number,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[invalid-label]",
                    original=None,
                    replacement=None,
                    description=(
                        f"{label} is not a PyTorch label "
                        "(please choose from https://github.com/pytorch/pytorch/labels)"
                    ),
                )
            )

        if label.startswith(("module:", "oncall:")) or label in ACCEPTABLE_OWNER_LABELS:
            continue

        lint_messages.append(
            LintMessage(
                path=filename,
                line=line_number,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[invalid-owner]",
                original=None,
                replacement=None,
                description=(
                    f"{label} is not an acceptable owner "
                    "(please update to another label or edit ACCEPTABLE_OWNERS_LABELS "
                    "in tools/linters/adapters/testowners_linter.py"
                ),
            )
        )

    return lint_messages


def check_file(filename: str) -> list[LintMessage]:
    lint_messages = []
    has_ownership_info = False

    with open(filename) as f:
        for idx, line in enumerate(f):
            if not line.startswith(OWNERS_PREFIX):
                continue

            has_ownership_info = True
            labels = json.loads(line[len(OWNERS_PREFIX) :])
            lint_messages.extend(check_labels(labels, filename, idx + 1))

    if has_ownership_info is False:
        lint_messages.append(
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[no-owner-info]",
                original=None,
                replacement=None,
                description="Missing a comment header with ownership information.",
            )
        )

    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="test ownership linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()
    lint_messages = []

    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
