#!/usr/bin/env python3
"""Ensure test classes declare a valid ``hw_classification`` attribute.

A JSON allowlist tracks test files that have *not yet* been fully classified.
Files in the allowlist are skipped silently.  Files **not** in the allowlist
must have ``hw_classification`` on every ``TestCase`` subclass — missing or
invalid values are reported as ERROR.

To graduate a file from the allowlist: add ``hw_classification`` to all test
classes in that file and remove its entry from the JSON file.
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "HW_CLASSIFICATION"
HW_CLASSIFICATION_ATTR = "hw_classification"  # class attribute name to check
HW_CLASSIFICATION_ENUM_CLASS = (
    "HardwareClassification"  # enum class the attribute must reference
)
INSTANTIATE_FN_NAME = "instantiate_device_type_tests"

GENERIC = "GENERIC"
ACCELERATOR = "ACCELERATOR"
CPU = "CPU"
CUDA = "CUDA"
MPS = "MPS"
XPU = "XPU"


# Files in this allowlist are temporarily excluded from hw_classification checks
ALLOWLIST_PATH = Path(__file__).resolve().parent / "hw_classification_allowlist.json"


def _load_allowlist() -> set[str]:
    if ALLOWLIST_PATH.exists():
        with open(ALLOWLIST_PATH) as f:
            return set(json.load(f))
    return set()


_allowlist: set[str] = _load_allowlist()


# Lint message types
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str
    line: int
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def create_error_msg(filename: str, line: int, description: str) -> LintMessage:
    return LintMessage(
        path=filename,
        line=line,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ERROR,
        name=f"[{HW_CLASSIFICATION_ATTR}]",
        original=None,
        replacement=None,
        description=description,
    )


def _is_test_file(filename: str) -> bool:
    name = os.path.basename(filename)
    if not name.endswith(".py"):
        return False
    return name.startswith("test_") or name.endswith("_test.py")


def _is_test_class(node: ast.ClassDef) -> bool:
    # Identify test classes by the presence of test_* methods rather than
    # inheritance. Resolving TestCase inheritance through AST is incomplete
    # because subclasses can be defined indirectly (e.g. NNTestCase, JitTestCase).
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_"):
            return True
    return False


def _find_hw_classification_assignment(
    node: ast.ClassDef,
) -> ast.Assign | ast.AnnAssign | None:
    """Find the class-level ``hw_classification`` assignment node, if present."""
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == HW_CLASSIFICATION_ATTR:
                    return stmt
        if isinstance(stmt, ast.AnnAssign):
            if (
                isinstance(stmt.target, ast.Name)
                and stmt.target.id == HW_CLASSIFICATION_ATTR
            ):
                return stmt
    return None


def _extract_hw_classification_value(assign_node: ast.AST) -> str | None:
    """Return the hardware classification name if the assignment uses a valid enum value."""
    value = None
    if isinstance(assign_node, ast.Assign):
        value = assign_node.value
    elif isinstance(assign_node, ast.AnnAssign) and assign_node.value is not None:
        value = assign_node.value

    if value is None:
        return None

    if (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == HW_CLASSIFICATION_ENUM_CLASS
    ):
        return value.attr

    return None


def _check_instantiation_in_tree(tree: ast.Module, class_name: str) -> bool:
    """Scan module-level statements for ``instantiate_device_type_tests(ClassName, ...)``."""
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == INSTANTIATE_FN_NAME:
                if (
                    call.args
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == class_name
                ):
                    return True
    return False


@dataclass
class RuleContext:
    """Context passed to each rule function during classification checks."""

    filename: str
    rel_path: str
    class_node: ast.ClassDef
    tree: ast.Module
    classification: str


RuleFunc = Callable[[RuleContext], list[LintMessage]]
rules: dict[str, list[RuleFunc]] = {}


def _register(*groups: str) -> Callable[[RuleFunc], RuleFunc]:
    """Decorator: register a rule function into one or more classification groups.

    Example::

        @_register(ACCELERATOR)
        def check_method_params(ctx): ...


        @_register(GENERIC, CPU, CUDA)
        def check_no_device_param(ctx): ...
    """

    def decorator(fn: RuleFunc) -> RuleFunc:
        for g in groups:
            rules.setdefault(g, []).append(fn)
        return fn

    return decorator


def _collect_params(stmt: ast.FunctionDef) -> set[str]:
    return {
        a.arg for a in stmt.args.args + stmt.args.posonlyargs + stmt.args.kwonlyargs
    }


@_register(GENERIC, CPU, CUDA, MPS, XPU)
def _check_no_device_param(ctx: RuleContext) -> list[LintMessage]:
    """Non-accelerator classes: test methods must not accept device/devices."""
    messages: list[LintMessage] = []
    for stmt in ctx.class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_"):
            params = _collect_params(stmt)
            if "device" in params or "devices" in params:
                messages.append(
                    create_error_msg(
                        ctx.filename,
                        stmt.lineno,
                        f"{ctx.classification} test method '{ctx.class_node.name}.{stmt.name}' "
                        f"must not accept a 'device' or 'devices' parameter.",
                    )
                )
    return messages


@_register(ACCELERATOR)
def _check_has_device_param(ctx: RuleContext) -> list[LintMessage]:
    """ACCELERATOR classes: every test_* method must accept device/devices."""
    messages: list[LintMessage] = []
    for stmt in ctx.class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_"):
            params = _collect_params(stmt)
            if "device" not in params and "devices" not in params:
                messages.append(
                    create_error_msg(
                        ctx.filename,
                        stmt.lineno,
                        f"{ctx.classification} test method '{ctx.class_node.name}.{stmt.name}' "
                        f"must accept a 'device' or 'devices' parameter.",
                    )
                )
    return messages


@_register(ACCELERATOR)
def _check_no_only_decorators(ctx: RuleContext) -> list[LintMessage]:
    """ACCELERATOR classes: test methods must not use only* decorators except onlyAccelerator."""
    messages: list[LintMessage] = []
    for stmt in ctx.class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_"):
            for dec in stmt.decorator_list:
                name = None
                if isinstance(dec, ast.Name):
                    name = dec.id
                elif isinstance(dec, ast.Attribute):
                    name = dec.attr
                if (
                    name is not None
                    and name.startswith("only")
                    and name != "onlyAccelerator"
                ):
                    messages.append(
                        create_error_msg(
                            ctx.filename,
                            stmt.lineno,
                            f"{ctx.classification} test method '{ctx.class_node.name}.{stmt.name}' "
                            f"must not use '@{name}' decorators except onlyAccelerator",
                        )
                    )
    return messages


@_register(ACCELERATOR)
def _check_must_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    if not _check_instantiation_in_tree(ctx.tree, ctx.class_node.name):
        return [
            create_error_msg(
                ctx.filename,
                ctx.class_node.lineno,
                f"{ctx.classification} class '{ctx.class_node.name}' must be "
                f"instantiated via 'instantiate_device_type_tests'.",
            )
        ]
    return []


@_register(GENERIC, CPU, CUDA, MPS, XPU)
def _check_must_not_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    if _check_instantiation_in_tree(ctx.tree, ctx.class_node.name):
        return [
            create_error_msg(
                ctx.filename,
                ctx.class_node.lineno,
                f"{ctx.classification} class '{ctx.class_node.name}' must not be "
                f"instantiated via 'instantiate_device_type_tests'.",
            )
        ]
    return []


def check_file(filename: str) -> list[LintMessage]:
    if not _is_test_file(filename):
        return []

    rel_path = os.path.relpath(filename)

    # Skip checks for files in the allowlist
    if rel_path in _allowlist:
        return []

    try:
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filename)
    except (OSError, SyntaxError) as e:
        raise RuntimeError(f"Failed to load Python source '{filename}'") from e

    messages: list[LintMessage] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _is_test_class(node):
            continue

        assign_node = _find_hw_classification_assignment(node)

        # Missing hw_classification attribute
        if assign_node is None:
            messages.append(
                create_error_msg(
                    filename,
                    node.lineno,
                    f"Test class '{node.name}' must declare "
                    f"{HW_CLASSIFICATION_ATTR} = HardwareClassification.<MEMBER>.",
                )
            )
            continue

        value = _extract_hw_classification_value(assign_node)

        # Value is not HardwareClassification.enum_member
        if value is None:
            messages.append(
                create_error_msg(
                    filename,
                    assign_node.lineno,
                    f"Could not determine {HW_CLASSIFICATION_ATTR} value for class '{node.name}'. "
                    f"Use 'HardwareClassification.<MEMBER>'.",
                )
            )
            continue

        # Dispatch to registered rule functions for this classification
        ctx = RuleContext(
            filename=filename,
            rel_path=rel_path,
            class_node=node,
            tree=tree,
            classification=value,
        )
        for rule in rules.get(value, []):
            messages.extend(rule(ctx))

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure test classes declare hw_classification.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    with mp.Pool(8) as pool:
        results = pool.map(check_file, args.filenames)

    for msgs in results:
        for msg in msgs:
            print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
