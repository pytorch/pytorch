#!/usr/bin/env python3
"""
This lint verifies that every Python test file (file that matches test_*.py or
*_test.py in the test folder) has a cuda hard code in `requires_gpu()`
decorated function to ensure that the test not fail on other GPU.

"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "TEST_DEVICE_BIAS"


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


class DeviceBiasVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.lint_messages: list[LintMessage] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check if the function is decorated with @requires_gpu, which indicates
        # that the function is intended to run on GPU devices (e.g., CUDA or XPU),
        # but ensure it does not hardcode the device to CUDA.
        has_requires_gpu = any(
            isinstance(d, ast.Call)
            and (
                (isinstance(d.func, ast.Name) and d.func.id == "requires_gpu")
                or (isinstance(d.func, ast.Attribute) and d.func.attr == "requires_gpu")
            )
            for d in node.decorator_list
        )
        # Detect the following pattern:
        # * device="cuda"
        # * device=torch.device("cuda")
        # * .cuda() call
        if has_requires_gpu:
            msg_prefix = "`@requires_gpu()` function should not hardcode"
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.keyword):
                    if subnode.arg == "device":
                        val = subnode.value
                        if isinstance(val, ast.Constant) and val.value == "cuda":
                            self.record(subnode, f"{msg_prefix} device='cuda'")
                        elif isinstance(val, ast.Call):
                            if (
                                isinstance(val.func, ast.Attribute)
                                and val.func.attr == "device"
                                and len(val.args) > 0
                                and isinstance(val.args[0], ast.Constant)
                                and val.args[0].value == "cuda"
                            ):
                                self.record(val, f"{msg_prefix} torch.device('cuda')")
                if isinstance(subnode, ast.Call) and isinstance(
                    subnode.func, ast.Attribute
                ):
                    if subnode.func.attr == "cuda":
                        self.record(subnode, f"{msg_prefix} .cuda() call")

        self.generic_visit(node)

    def record(self, node: ast.AST, message: str) -> None:
        self.lint_messages.append(
            LintMessage(
                path=self.filename,
                line=getattr(node, "lineno", None),
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[device-bias]",
                original=None,
                replacement=None,
                description=message,
            )
        )


def check_file(filename: str) -> list[LintMessage]:
    with open(filename) as f:
        source = f.read()
        tree = ast.parse(source, filename=filename)
        checker = DeviceBiasVisitor(filename)
        checker.visit(tree)

    return checker.lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="test device bias code",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    with mp.Pool(8) as pool:
        lint_messages = pool.map(check_file, args.filenames)

    flat_lint_messages = []
    for sublist in lint_messages:
        flat_lint_messages.extend(sublist)

    for lint_message in flat_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
