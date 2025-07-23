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


DEVICE_BIAS = ["cuda", "xpu", "mps"]


class DeviceBiasVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.lint_messages: list[LintMessage] = []

    def _has_requires_gpu_decorator(self, node: ast.FunctionDef) -> bool:
        for d in node.decorator_list:
            if isinstance(d, ast.Name) and d.id == "requires_gpu":
                return True
            if (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "requires_gpu"
            ):
                return True
        return False

    # check device = "cuda" or torch.device("cuda")
    def _check_keyword_device(self, subnode: ast.keyword, msg_prefix: str) -> None:
        if subnode.arg != "device":
            return

        val = subnode.value
        if isinstance(val, ast.Constant) and any(
            bias in val.value for bias in DEVICE_BIAS
        ):
            self.record(
                subnode,
                f"{msg_prefix} device='{val.value}', suggest to use device=GPU_TYPE",
            )
        elif isinstance(val, ast.Call):
            if (
                isinstance(val.func, ast.Attribute)
                and val.func.attr == "device"
                and len(val.args) > 0
                and isinstance(val.args[0], ast.Constant)
                and any(bias in val.args[0].value for bias in DEVICE_BIAS)
            ):
                self.record(
                    val,
                    f"{msg_prefix} torch.device('{val.args[0].value}'), suggest to use torch.device(GPU_TYPE)",
                )

    # check .cuda() or .to("cuda")
    def _check_device_methods(self, subnode: ast.Call, msg_prefix: str) -> None:
        func = subnode.func
        if not isinstance(func, ast.Attribute):
            return
        method_name = func.attr
        if method_name in DEVICE_BIAS:
            self.record(
                subnode,
                f"{msg_prefix} .{method_name}(), suggest to use .to(GPU_TYPE)",
            )
        elif method_name == "to" and subnode.args:
            arg = subnode.args[0]
            if isinstance(arg, ast.Constant) and any(
                bias in arg.value for bias in DEVICE_BIAS
            ):
                self.record(
                    subnode,
                    f"{msg_prefix} .to('{arg.value}'), suggest to use .to(GPU_TYPE)",
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check if the function is decorated with @requires_gpu, which indicates
        # that the function is intended to run on GPU devices (e.g., CUDA or XPU),
        # but ensure it does not hardcode the device to CUDA.
        if not self._has_requires_gpu_decorator(node):
            self.generic_visit(node)
            return

        msg_prefix = "`@requires_gpu` function should not hardcode"
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.keyword):
                self._check_keyword_device(subnode, msg_prefix)
            elif isinstance(subnode, ast.Call) and isinstance(
                subnode.func, ast.Attribute
            ):
                self._check_device_methods(subnode, msg_prefix)

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
        description="Detect Device bias in python functions decorated with [require_gpu]"
        " that may potentially break support for other GPU devices.",
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
