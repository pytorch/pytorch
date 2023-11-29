#!/usr/bin/env python3
"""
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

This lint verifies that every Python test file (file that matches test_*.py or *_test.py in the test folder)
has valid ownership information in a comment header. Valid means:
  - The format of the header follows the pattern "# Owner(s): ["list", "of owner", "labels"]
  - Each owner label actually exists in PyTorch
  - Each owner label starts with "module: " or "oncall: " or is in ACCEPTABLE_OWNER_LABELS
"""
import argparse
import json
from enum import Enum
from typing import List, NamedTuple, Optional
import multiprocessing as mp
import libcst as cst
import libcst.matchers as m

LINTER_CODE = "TEST_HAS_MAIN"


class TypingCollector(cst.CSTVisitor):
    def __init__(self, raise_or_run: str = "any"):
        super().__init__()
        self.found = False
        self._docstring: Optional[str] = None
        self.raise_or_run = raise_or_run

    def visit_Module(self, node: cst.Module) -> bool:
        name = m.Name("__name__")
        main = m.SimpleString('"__main__"') | m.SimpleString("'__main__'")
        run_test_call = m.Call(func=m.Name("run_tests") | m.Attribute(attr=m.Name("run_tests")))
        raise_block = m.Raise()

        # left = node.test
        s1 = m.Comparison(
            name,
            [m.ComparisonTarget(m.Equal(), main)],
        )
        s2 = m.Comparison(
            main,
            [m.ComparisonTarget(m.Equal(), name)],
        )
        for child in node.children:
            if m.matches(child, m.If(test=s1 | s2)):
                to_match = raise_block if self.raise_or_run == "raise" else run_test_call | raise_block
                if len(m.findall(node, to_match)) > 0:
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

SHOULD_RAISE = [
    "jit", "quantization", "ao/sparsity",
]

def check_file(filename: str) -> List[LintMessage]:
    lint_messages = []

    with open(filename) as f:
        file = f.read()
        should_raise = any(f"test/{folder}" in filename for folder in SHOULD_RAISE)
        v = TypingCollector(raise_or_run="raise" if should_raise else "any")
        cst.parse_module(file).visit(v)
        if v.found == False:
            message = "Needs to have a main block which raises an exception or calls run_tests"
            if should_raise:
                message = "Needs to have a main block which raises an exception"
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=len(file.rea()),
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[no-main]",
                    original=None,
                    replacement=None,
                    description="message",
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
