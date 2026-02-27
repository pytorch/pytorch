#!/usr/bin/env python3
"""
This lint verifies that every Python test file (file that matches test_*.py or
*_test.py in the test folder) has a cuda hard code in `requires_gpu()` or
`requires_triton()` decorated function or `if HAS_GPU:` guarded main section,
to ensure that the test not fail on other GPU devices.
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
GPU_RELATED_DECORATORS = {"requires_gpu", "requires_triton"}


def is_main_has_gpu(tree: ast.AST) -> bool:
    def _contains_has_gpu(node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id in ["HAS_GPU", "RUN_GPU"]:
            return True
        elif isinstance(node, ast.BoolOp):
            return any(_contains_has_gpu(value) for value in node.values)
        elif isinstance(node, ast.UnaryOp):
            return _contains_has_gpu(node.operand)
        elif isinstance(node, ast.Compare):
            return _contains_has_gpu(node.left) or any(
                _contains_has_gpu(comp) for comp in node.comparators
            )
        elif isinstance(node, (ast.IfExp, ast.Call)):
            return False
        return False

    for node in ast.walk(tree):
        # Detect if __name__ == "__main__":
        if isinstance(node, ast.If):
            if (
                isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                if any(
                    isinstance(comp, ast.Constant) and comp.value == "__main__"
                    for comp in node.test.comparators
                ):
                    for inner_node in node.body:
                        if isinstance(inner_node, ast.If) and _contains_has_gpu(
                            inner_node.test
                        ):
                            return True
    return False


class DeviceBiasVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, is_gpu_test_suite: bool) -> None:
        self.filename = filename
        self.lint_messages: list[LintMessage] = []
        self.is_gpu_test_suite = is_gpu_test_suite

    def _has_proper_decorator(self, node: ast.FunctionDef) -> bool:
        for d in node.decorator_list:
            if isinstance(d, ast.Name) and d.id in GPU_RELATED_DECORATORS:
                return True
            if (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id in GPU_RELATED_DECORATORS
            ):
                return True
        return False

    # check device = "cuda" or torch.device("cuda")
    def _check_keyword_device(self, subnode: ast.keyword, msg_prefix: str) -> None:
        if subnode.arg != "device":
            return
        val = subnode.value
        if isinstance(val, ast.Constant) and any(
            # pyrefly: ignore [not-iterable, unsupported-operation]
            bias in val.value
            for bias in DEVICE_BIAS
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
                # pyrefly: ignore [not-iterable, unsupported-operation]
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
                # pyrefly: ignore [not-iterable, unsupported-operation]
                bias in arg.value
                for bias in DEVICE_BIAS
            ):
                self.record(
                    subnode,
                    f"{msg_prefix} .to('{arg.value}'), suggest to use .to(GPU_TYPE)",
                )

    def _check_with_statement(self, node: ast.With, msg_prefix: str) -> None:
        for item in node.items:
            ctx_expr = item.context_expr
            if isinstance(ctx_expr, ast.Call):
                func = ctx_expr.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "device"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                    and ctx_expr.args
                    and isinstance(ctx_expr.args[0], ast.Constant)
                    # pyrefly: ignore [not-iterable, unsupported-operation]
                    and any(bias in ctx_expr.args[0].value for bias in DEVICE_BIAS)
                ):
                    self.record(
                        ctx_expr,
                        f"{msg_prefix} `with torch.device('{ctx_expr.args[0].value}')`, suggest to use torch.device(GPU_TYPE)",
                    )

    def _check_node(self, node: ast.AST, msg_prefix: str) -> None:
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.keyword):
                self._check_keyword_device(subnode, msg_prefix)
            elif isinstance(subnode, ast.Call) and isinstance(
                subnode.func, ast.Attribute
            ):
                self._check_device_methods(subnode, msg_prefix)
            elif isinstance(subnode, ast.With):
                self._check_with_statement(subnode, msg_prefix)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._has_proper_decorator(node):
            msg_prefix = (
                "`@requires_gpu` or `@requires_triton` function should not hardcode"
            )
            self._check_node(node, msg_prefix)
        elif self.is_gpu_test_suite:
            # If the function is guarded by HAS_GPU in main(), we still need to check for device bias
            msg_prefix = "The test suites is shared amount GPUS, should not hardcode"
            self._check_node(node, msg_prefix)
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
        is_gpu_test_suite = is_main_has_gpu(tree)
        checker = DeviceBiasVisitor(filename, is_gpu_test_suite)
        checker.visit(tree)
    return checker.lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect Device bias in functions decorated with requires_gpu/requires_triton"
        " or guarded by HAS_GPU block in main() that may break other GPU devices.",
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
