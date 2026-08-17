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
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "TEST_LINTER"
HW_CLASSIFICATION_ATTR = "hw_classification"
INSTANTIATE_FN_NAME = "instantiate_device_type_tests"
REPO_ROOT = Path(__file__).resolve().parents[3]

_KWARG_UNKNOWN = object()  # sentinel: kwarg present but not a literal


# Mirrors the member names of `torch.testing._internal.common_utils.HardwareClassification`.
# Values differ from upstream; only member names are used for matching.
# Defined locally to avoid importing test infrastructure into the linter.
class HardwareClassification(Enum):
    GENERIC = "GENERIC"
    ACCELERATOR = "ACCELERATOR"
    CPU = "CPU"
    CUDA = "CUDA"
    MPS = "MPS"
    XPU = "XPU"


DEVICE_SPECIFIC_CLASSIFICATIONS = {
    HardwareClassification.CPU,
    HardwareClassification.CUDA,
    HardwareClassification.MPS,
    HardwareClassification.XPU,
}

# Files in this allowlist are temporarily excluded from test linter checks
ALLOWLIST_PATH = Path(__file__).resolve().parent / "test_linter_allowlist.json"


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


error_msg = partial(
    LintMessage,
    char=None,
    code=LINTER_CODE,
    severity=LintSeverity.ERROR,
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


def _get_hw_classification(
    node: ast.ClassDef,
) -> HardwareClassification | None:
    """Parse the `hw_classification` attribute from *node*.

    The value is returned as a `HardwareClassification` enum member.

    Only accepts the exact forms:

        hw_classification = HardwareClassification.<MEMBER>
        hw_classification: HardwareClassification = HardwareClassification.<MEMBER>

    Returns `None` if the attribute is absent or does not match one of the
    supported forms.
    """
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not (
                isinstance(target, ast.Name) and target.id == HW_CLASSIFICATION_ATTR
            ):
                continue
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if not (
                isinstance(target, ast.Name) and target.id == HW_CLASSIFICATION_ATTR
            ):
                continue
            if not (
                isinstance(stmt.annotation, ast.Name)
                and stmt.annotation.id == HardwareClassification.__name__
                and stmt.value is not None
            ):
                return None
            value = stmt.value
        else:
            continue

        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == HardwareClassification.__name__
        ):
            try:
                return HardwareClassification[value.attr]
            except KeyError:
                return None

        return None

    return None


def _get_instantiation_call(tree: ast.Module, class_name: str) -> ast.Call | None:
    """Find a top-level instantiate_device_type_tests call for a class."""
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


def _get_call_kwarg_value(
    call: ast.Call | None, param_name: str
) -> list[str] | None | object:
    """Return statically known string list value of a keyword argument.

    Returns:
        - None: keyword argument is absent.
        - list[str]: keyword argument is a statically known string list.
        - _KWARG_UNKNOWN: keyword argument exists but cannot be statically resolved.
    """
    if call is None:
        return None

    for kw_item in call.keywords:
        if kw_item.arg != param_name:
            continue

        node = kw_item.value

        if isinstance(node, ast.Constant) and node.value is None:
            return None

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]

        if isinstance(node, (ast.List, ast.Tuple)):
            result = [
                elt.value
                for elt in node.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]

            # list contains non-string elements
            if len(result) != len(node.elts):
                return _KWARG_UNKNOWN
            return result

        return _KWARG_UNKNOWN

    return None


@dataclass
class InstantiationContext:
    """Context for an ``instantiate_device_type_tests`` call."""

    call: ast.Call
    only_for: list[str] | None | object = None
    except_for: list[str] | None | object = None


@dataclass
class RuleContext:
    """Context passed to each rule function during test class linter checks."""

    filename: str
    class_node: ast.ClassDef
    classification: HardwareClassification
    test_methods: list[ast.FunctionDef] = field(default_factory=list)
    instantiation: InstantiationContext | None = None

    @classmethod
    def from_node(
        cls,
        filename: str,
        class_node: ast.ClassDef,
        tree: ast.Module,
        classification: HardwareClassification,
    ) -> RuleContext:
        test_methods = [
            stmt
            for stmt in class_node.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_")
        ]

        instantiation = None
        call = _get_instantiation_call(tree, class_node.name)
        if call is not None:
            instantiation = InstantiationContext(
                call=call,
                only_for=_get_call_kwarg_value(call, "only_for"),
                except_for=_get_call_kwarg_value(call, "except_for"),
            )
        return cls(
            filename=filename,
            class_node=class_node,
            classification=classification,
            test_methods=test_methods,
            instantiation=instantiation,
        )


RuleFunc = Callable[[RuleContext], list[LintMessage]]
rules: dict[HardwareClassification, list[RuleFunc]] = {}


def _register(*groups: HardwareClassification) -> Callable[[RuleFunc], RuleFunc]:
    """Decorator: register a rule function into one or more classification groups.

    Example::

        @_register(HardwareClassification.ACCELERATOR)
        def _check_no_only_for(ctx): ...


        @_register(HardwareClassification.GENERIC)
        def _check_no_device_param(ctx): ...
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
            error_msg(
                name="[device_param]",
                path=ctx.filename,
                line=stmt.lineno,
                description=f"{ctx.classification.value} test method '{ctx.class_node.name}.{stmt.name}' "
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
    is_instantiated = ctx.instantiation is not None
    if is_instantiated == required:
        return []
    action = "must be" if required else "must not be"
    return [
        error_msg(
            name="[instantiation]",
            path=ctx.filename,
            line=ctx.class_node.lineno,
            description=f"{ctx.classification.value} class '{ctx.class_node.name}' {action} "
            f"instantiated via 'instantiate_device_type_tests'.",
        )
    ]


# ---------------------------------------------------------------------------
# GENERIC rules
# ---------------------------------------------------------------------------


@_register(HardwareClassification.GENERIC)
def _check_must_not_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    return _check_instantiation(ctx, required=False)


@_register(HardwareClassification.GENERIC)
def _check_no_device_param(ctx: RuleContext) -> list[LintMessage]:
    return _check_device_param(ctx, required=False)


# ---------------------------------------------------------------------------
# ACCELERATOR rules
# ---------------------------------------------------------------------------


@_register(HardwareClassification.ACCELERATOR)
def _check_accelerator_must_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    return _check_instantiation(ctx, required=True)


@_register(HardwareClassification.ACCELERATOR)
def _check_accelerator_has_device_param(ctx: RuleContext) -> list[LintMessage]:
    return _check_device_param(ctx, required=True)


@_register(HardwareClassification.ACCELERATOR)
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
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    name = dec.func.id
                elif isinstance(dec.func, ast.Attribute):
                    name = dec.func.attr
            if (
                name is not None
                and name.startswith("only")
                and name != "onlyAccelerator"
            ):
                messages.append(
                    error_msg(
                        name="[decorator]",
                        path=ctx.filename,
                        line=stmt.lineno,
                        description=f"{ctx.classification.value} test method '{ctx.class_node.name}.{stmt.name}' "
                        f"must not use '@{name}' decorators except onlyAccelerator",
                    )
                )
    return messages


@_register(HardwareClassification.ACCELERATOR)
def _check_no_only_for(ctx: RuleContext) -> list[LintMessage]:
    """ACCELERATOR classes: instantiate_device_type_tests must not use only_for.

    Use except_for for a blacklist approach instead.
    """
    if ctx.instantiation is not None and ctx.instantiation.only_for is not None:
        return [
            error_msg(
                name="[only_for]",
                path=ctx.filename,
                line=ctx.instantiation.call.lineno,
                description=f"{ctx.classification.value} class '{ctx.class_node.name}' "
                f"must not use only_for in instantiate_device_type_tests. "
                f"Use except_for instead (blacklist approach).",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# CPU / CUDA / MPS / XPU (device-specific) rules
# ---------------------------------------------------------------------------


@_register(*DEVICE_SPECIFIC_CLASSIFICATIONS)
def _check_device_specific_must_be_instantiated(ctx: RuleContext) -> list[LintMessage]:
    return _check_instantiation(ctx, required=True)


@_register(*DEVICE_SPECIFIC_CLASSIFICATIONS)
def _check_device_specific_has_device_param(ctx: RuleContext) -> list[LintMessage]:
    return _check_device_param(ctx, required=True)


@_register(*DEVICE_SPECIFIC_CLASSIFICATIONS)
def _check_no_except_for(ctx: RuleContext) -> list[LintMessage]:
    """Device-specific classes: instantiate_device_type_tests must not use except_for."""
    if ctx.instantiation is not None and ctx.instantiation.except_for is not None:
        return [
            error_msg(
                name="[except_for]",
                path=ctx.filename,
                line=ctx.instantiation.call.lineno,
                description=f"{ctx.classification.value} class '{ctx.class_node.name}' "
                f"must not use except_for in instantiate_device_type_tests.",
            )
        ]
    return []


@_register(*DEVICE_SPECIFIC_CLASSIFICATIONS)
def _check_only_for_matches_device(ctx: RuleContext) -> list[LintMessage]:
    """Device-specific classes: instantiate_device_type_tests must specify
    only_for matching exactly the class's classification."""
    if ctx.instantiation is None:
        return []
    expected = ctx.classification.value.lower()

    if (
        ctx.instantiation.only_for is None
        or ctx.instantiation.only_for is _KWARG_UNKNOWN
    ):
        return [
            error_msg(
                name="[only_for]",
                path=ctx.filename,
                line=ctx.instantiation.call.lineno,
                description=f"{ctx.classification.value} class '{ctx.class_node.name}' "
                f"must use only_for='{expected}' "
                f"in instantiate_device_type_tests.",
            )
        ]
    if ctx.instantiation.only_for != [expected]:
        return [
            error_msg(
                name="[only_for]",
                path=ctx.filename,
                line=ctx.instantiation.call.lineno,
                description=f"{ctx.classification.value} class '{ctx.class_node.name}' "
                f"has only_for values {ctx.instantiation.only_for}, "
                f"but must be exactly {[expected]}.",
            )
        ]
    return []


def check_file(filename: str) -> list[LintMessage]:
    if not _is_test_file(filename):
        return []

    rel_path = os.path.relpath(filename, REPO_ROOT).replace("\\", "/")

    # Skip checks for files in the allowlist
    if rel_path in _allowlist:
        return []

    try:
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filename)
    except (OSError, SyntaxError) as e:
        return [
            error_msg(
                name="[parse_error]",
                path=filename,
                line=0,
                description=f"Failed to parse '{filename}': {e}",
            )
        ]

    messages: list[LintMessage] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _is_test_class(node):
            continue

        classification = _get_hw_classification(node)
        if classification is None:
            messages.append(
                error_msg(
                    name="[hw_classification]",
                    path=filename,
                    line=node.lineno,
                    description=f"Test class '{node.name}' is missing or has an invalid "
                    f"hw_classification. Only the exact forms below are accepted "
                    f"(aliased imports are not recognized):\n"
                    f"    hw_classification = HardwareClassification.<MEMBER>\n"
                    f"    hw_classification: HardwareClassification = HardwareClassification.<MEMBER>",
                )
            )
            continue

        # Dispatch to registered rule functions for this classification
        ctx = RuleContext.from_node(
            filename=filename,
            class_node=node,
            tree=tree,
            classification=classification,
        )
        for rule in rules.get(classification, []):
            messages.extend(rule(ctx))

    return messages


def _default_num_workers() -> int | None:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit() and int(max_jobs) > 0:
        return int(max_jobs)
    return os.cpu_count()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure test classes declare hw_classification.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=_default_num_workers(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                print(
                    json.dumps(
                        error_msg(
                            name="[internal_error]",
                            path=futures[future],
                            line=0,
                            description=f"Linter failed on '{futures[future]}'",
                        )._asdict()
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
