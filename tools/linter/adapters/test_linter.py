#!/usr/bin/env python3
"""Validate test case requirements.

This linter enforces test case requirements, including hardware classification
declaration, test instantiation patterns, test method signatures, and
supported decorators.

A JSON allowlist tracks test files that are not yet migrated to this linter's
requirements. Files in the allowlist are skipped silently. Files not in the
allowlist must satisfy the test case requirements defined below.

All test classes inheriting from `TestCase` must first declare a valid
`hw_classification` attribute. Supported values are `GENERIC`, `ACCELERATOR`,
`CPU`, `CUDA`, `MPS`, and `XPU`.

The requirements for each `hw_classification` are summarized below:

  GENERIC
    - Class must not be used with instantiate_device_type_tests.
    - Test methods must not accept device/devices parameter.

  ACCELERATOR
    - Class must be used with instantiate_device_type_tests.
    - Every test method must accept device/devices parameter.
    - Test methods must not use @only* decorators (except @onlyAccelerator).
    - instantiate_device_type_tests must not use only_for (use except_for
      as a blacklist approach instead).

  CPU / CUDA / MPS / XPU (device-specific)
    - Class must be used with instantiate_device_type_tests.
    - Every test method must accept device/devices parameter.
    - instantiate_device_type_tests must use only_for matching the device
      (e.g. only_for='cuda').
    - instantiate_device_type_tests must not use except_for.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "TEST_LINTER"
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


_error_msg = partial(
    LintMessage,
    char=None,
    code=LINTER_CODE,
    severity=LintSeverity.ERROR,
    name=f"[{HW_CLASSIFICATION_ATTR}]",
    original=None,
    replacement=None,
)


def _is_test_file(filename: str) -> bool:
    name = os.path.basename(filename)
    if not name.endswith(".py"):
        return False
    return name.startswith("test_") or name.endswith("_test.py")


def _is_test_class(node: ast.ClassDef) -> bool:
    # Identify test classes by the presence of test_* methods rather than
    # inheritance. Resolving TestCase inheritance through AST is incomplete
    # because subclasses can be defined indirectly (e.g. NNTestCase, JitTestCase)
    # and across different files. Therefore, intermediate base classes and
    # concrete test classes are treated uniformly: any class defining test_*
    # methods must declare hw_classification.
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


def _get_instantiation_call(tree: ast.Module, class_name: str) -> ast.Call | None:
    """Find the ``instantiate_device_type_tests(ClassName, ...)`` call node."""
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name) and call.func.id == INSTANTIATE_FN_NAME:
                if (
                    call.args
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == class_name
                ):
                    return call
    return None


def _get_call_kwarg_value(call: ast.Call | None, param_name: str) -> list[str] | None:
    """Return the value list of *param_name* from the call, or None if absent or empty."""
    if call is None:
        return None
    kw = None
    for kw_item in call.keywords:
        if kw_item.arg == param_name:
            kw = kw_item
            break
    if kw is None:
        return None

    node = kw.value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None

    result = [
        elt.value
        for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]
    return result if result else None


@dataclass
class RuleContext:
    """Context passed to each rule function during classification checks."""

    filename: str
    rel_path: str
    tree: ast.Module  # AST of the entire Python file.
    class_node: ast.ClassDef  # AST node of the test class being checked.
    classification: str  # Hardware classification of the test class.

    # instantiate_device_type_tests call information
    instantiation_call: ast.Call | None = (
        None  # AST call node for instantiate_device_type_tests, if present.
    )
    only_for: list[str] | None = None
    except_for: list[str] | None = None

    # test_* method AST nodes
    test_methods: list[ast.FunctionDef] = field(default_factory=list)

    @classmethod
    def from_node(
        cls,
        filename: str,
        rel_path: str,
        class_node: ast.ClassDef,
        tree: ast.Module,
        classification: str,
    ) -> RuleContext:
        call = _get_instantiation_call(tree, class_node.name)
        test_methods = [
            stmt
            for stmt in class_node.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_")
        ]
        return cls(
            filename=filename,
            rel_path=rel_path,
            class_node=class_node,
            tree=tree,
            classification=classification,
            instantiation_call=call,
            only_for=_get_call_kwarg_value(call, "only_for"),
            except_for=_get_call_kwarg_value(call, "except_for"),
            test_methods=test_methods,
        )


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


def _check_device_param(ctx: RuleContext, *, required: bool) -> list[LintMessage]:
    """Validate whether test methods accept device(s) parameters."""
    messages: list[LintMessage] = []
    for stmt in ctx.test_methods:
        params = {
            a.arg for a in stmt.args.args + stmt.args.posonlyargs + stmt.args.kwonlyargs
        }
        has_device_param = "device" in params or "devices" in params
        if has_device_param == required:
            continue
        action = "must accept" if required else "must not accept"
        messages.append(
            _error_msg(
                path=ctx.filename,
                line=stmt.lineno,
                description=f"{ctx.classification} test method '{ctx.class_node.name}.{stmt.name}' "
                f"{action} a 'device' or 'devices' parameter.",
            )
        )
    return messages


def _check_instantiation(
    ctx: RuleContext,
    *,
    required: bool,
) -> list[LintMessage]:
    """Validate test class instantiation through `instantiate_device_type_tests`."""
    is_instantiated = ctx.instantiation_call is not None
    if is_instantiated == required:
        return []
    action = "must be" if required else "must not be"
    return [
        _error_msg(
            path=ctx.filename,
            line=ctx.class_node.lineno,
            description=f"{ctx.classification} class '{ctx.class_node.name}' {action} "
            f"instantiated via 'instantiate_device_type_tests'.",
        )
    ]


# ---------------------------------------------------------------------------
# GENERIC rules
# ---------------------------------------------------------------------------


@_register(GENERIC)
def _check_must_not_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    return _check_instantiation(ctx, required=False)


@_register(GENERIC)
def _check_no_device_param(ctx: RuleContext) -> list[LintMessage]:
    return _check_device_param(ctx, required=False)


# ---------------------------------------------------------------------------
# ACCELERATOR rules
# ---------------------------------------------------------------------------


@_register(ACCELERATOR)
def _check_must_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    return _check_instantiation(ctx, required=True)


@_register(ACCELERATOR)
def _check_has_device_param(ctx: RuleContext) -> list[LintMessage]:
    return _check_device_param(ctx, required=True)


@_register(ACCELERATOR)
def _check_no_only_decorators(ctx: RuleContext) -> list[LintMessage]:
    """ACCELERATOR classes: test methods must not use only* decorators except onlyAccelerator."""
    messages: list[LintMessage] = []
    for stmt in ctx.test_methods:
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
                    _error_msg(
                        path=ctx.filename,
                        line=stmt.lineno,
                        description=f"{ctx.classification} test method '{ctx.class_node.name}.{stmt.name}' "
                        f"must not use '@{name}' decorators except onlyAccelerator",
                    )
                )
    return messages


@_register(ACCELERATOR)
def _check_no_only_for(ctx: RuleContext) -> list[LintMessage]:
    """ACCELERATOR classes: instantiate_device_type_tests must not use only_for.

    Use except_for for a blacklist approach instead.
    """
    if ctx.instantiation_call is not None and ctx.only_for is not None:
        return [
            _error_msg(
                path=ctx.filename,
                line=ctx.instantiation_call.lineno,
                description=f"{ctx.classification} class '{ctx.class_node.name}' "
                f"must not use only_for in instantiate_device_type_tests. "
                f"Use except_for instead (blacklist approach).",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# CPU / CUDA / MPS / XPU (device-specific) rules
# ---------------------------------------------------------------------------


@_register(CPU, CUDA, MPS, XPU)
def _check_must_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    return _check_instantiation(ctx, required=True)


@_register(CPU, CUDA, MPS, XPU)
def _check_has_device_param(ctx: RuleContext) -> list[LintMessage]:
    return _check_device_param(ctx, required=True)


@_register(CPU, CUDA, MPS, XPU)
def _check_no_except_for(ctx: RuleContext) -> list[LintMessage]:
    """Device-specific classes: instantiate_device_type_tests must not use except_for."""
    if ctx.instantiation_call is not None and ctx.except_for is not None:
        return [
            _error_msg(
                path=ctx.filename,
                line=ctx.instantiation_call.lineno,
                description=f"{ctx.classification} class '{ctx.class_node.name}' "
                f"must not use except_for in instantiate_device_type_tests.",
            )
        ]
    return []


@_register(CPU, CUDA, MPS, XPU)
def _check_only_for_matches_device(ctx: RuleContext) -> list[LintMessage]:
    """Device-specific classes: instantiate_device_type_tests must specify
    only_for matching exactly the class's classification."""
    if ctx.instantiation_call is None:
        return []
    expected = ctx.classification.lower()

    if ctx.only_for is None:
        return [
            _error_msg(
                path=ctx.filename,
                line=ctx.instantiation_call.lineno,
                description=f"{ctx.classification} class '{ctx.class_node.name}' "
                f"must use only_for='{expected}' "
                f"in instantiate_device_type_tests.",
            )
        ]
    if ctx.only_for != [expected]:
        return [
            _error_msg(
                path=ctx.filename,
                line=ctx.instantiation_call.lineno,
                description=f"{ctx.classification} class '{ctx.class_node.name}' "
                f"has only_for values {ctx.only_for}, "
                f"but must be exactly {[expected]}.",
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
                _error_msg(
                    path=filename,
                    line=node.lineno,
                    description=f"Test class '{node.name}' must declare "
                    f"{HW_CLASSIFICATION_ATTR} = HardwareClassification.<MEMBER>.",
                )
            )
            continue

        value = _extract_hw_classification_value(assign_node)

        # Value is not HardwareClassification.enum_member
        if value is None:
            messages.append(
                _error_msg(
                    path=filename,
                    line=assign_node.lineno,
                    description=f"Could not determine {HW_CLASSIFICATION_ATTR} value for class '{node.name}'. "
                    f"Use 'HardwareClassification.<MEMBER>'.",
                )
            )
            continue

        # Dispatch to registered rule functions for this classification
        ctx = RuleContext.from_node(
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

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
