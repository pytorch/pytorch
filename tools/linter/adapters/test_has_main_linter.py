#!/usr/bin/env python3
"""
This lint verifies that every Python test file (file that matches test_*.py or
*_test.py in the test folder) has a main block which raises an exception or
calls run_tests to ensure that the test will be run in OSS CI.

Takes ~2 minuters to run without the multiprocessing, probably overkill.
"""
import argparse
import json
import multiprocessing as mp
from enum import Enum
from typing import List, NamedTuple, Optional

import libcst as cst
import libcst.matchers as m

LINTER_CODE = "TEST_HAS_MAIN"


class HasMainVisiter(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.found = False

    def visit_Module(self, node: cst.Module) -> bool:
        name = m.Name("__name__")
        main = m.SimpleString('"__main__"') | m.SimpleString("'__main__'")
        run_test_call = m.Call(
            func=m.Name("run_tests") | m.Attribute(attr=m.Name("run_tests"))
        )
        raise_block = m.Raise()

        # name == main or main == name
        if_main1 = m.Comparison(
            name,
            [m.ComparisonTarget(m.Equal(), main)],
        )
        if_main2 = m.Comparison(
            main,
            [m.ComparisonTarget(m.Equal(), name)],
        )
        for child in node.children:
            if m.matches(child, m.If(test=if_main1 | if_main2)):
                if m.findall(child, raise_block | run_test_call):
                    self.found = True
                    break

        return False


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


def check_file(filename: str) -> List[LintMessage]:
    lint_messages = []

    with open(filename) as f:
        file = f.read()
        v = HasMainVisiter()
        cst.parse_module(file).visit(v)
        if not v.found:
            message = (
                "Test files need to have a main block which either calls run_tests "
                + "(to ensure that the tests are run during OSS CI) or raises an exception "
                + "and added to the blocklist in test/run_test.py"
            )
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[no-main]",
                    original=None,
                    replacement=None,
                    description=message,
                )
            )
    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="test files should have main block linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    pool = mp.Pool(8)
    lint_messages = pool.map(check_file, args.filenames)
    pool.close()
    pool.join()

    flat_lint_messages = []
    for sublist in lint_messages:
        flat_lint_messages.extend(sublist)

    for lint_message in flat_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
