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
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "HW_CLASSIFICATION"
HW_CLASSIFICATION_ATTR = "hw_classification"  # class attribute name to check
HW_CLASSIFICATION_ENUM_CLASS = (
    "HardwareClassification"  # enum class the attribute must reference
)
INSTANTIATE_FN_NAME = "instantiate_device_type_tests"  # ACCELERATOR classes must be instantiated via this function
ACCELERATOR = "ACCELERATOR"  # classification requiring device parameter and instantiate_device_type_tests

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
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def _is_test_file(filename: str) -> bool:
    name = os.path.basename(filename)
    if not name.endswith(".py"):
        return False
    return name.startswith("test_") or name.endswith("_test.py")


def _is_test_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        # class TestSubclass(TestCase)
        if isinstance(base, ast.Name) and base.id == "TestCase":
            return True
        # class TestSubclass(common_utils.TestCase)
        if isinstance(base, ast.Attribute) and base.attr == "TestCase":
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


def _check_accelerator_methods(node: ast.ClassDef, filename: str) -> list[LintMessage]:
    """ACCELERATOR classes: every test_* method must accept device/devices."""
    messages: list[LintMessage] = []
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_"):
            params = (
                [a.arg for a in stmt.args.args]
                + [a.arg for a in stmt.args.posonlyargs]
                + [a.arg for a in stmt.args.kwonlyargs]
            )
            if "device" not in params and "devices" not in params:
                messages.append(
                    LintMessage(
                        path=filename,
                        line=stmt.lineno,
                        char=None,
                        code=LINTER_CODE,
                        severity=LintSeverity.ERROR,
                        name=f"[{HW_CLASSIFICATION_ATTR}]",
                        original=None,
                        replacement=None,
                        description=(
                            f"ACCELERATOR test method '{node.name}.{stmt.name}' "
                            f"must accept a 'device' or 'devices' parameter."
                        ),
                    )
                )
    return messages


# Sub-directories whose test classes are instantiated in a separate file.
CROSS_FILE_INSTANTIATION = {
    "test/ao/sparsity": "test/test_ao_sparsity.py",
    "test/quantization": "test/test_quantization.py",
}


def _class_is_instantiated(class_name: str, rel_path: str, tree: ast.Module) -> bool:
    """Check if *class_name* is passed to ``instantiate_device_type_tests``."""
    for prefix, agg_file in CROSS_FILE_INSTANTIATION.items():
        if rel_path.startswith(prefix):
            return _check_instantiation_in_file(agg_file, class_name)

    return _check_instantiation_in_tree(tree, class_name)


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


def _check_instantiation_in_file(filepath: str, class_name: str) -> bool:
    try:
        tree = ast.parse(Path(filepath).read_text(encoding="utf-8"), filename=filepath)
    except (OSError, SyntaxError) as e:
        raise RuntimeError(f"Failed to load Python source '{filepath}'") from e

    return _check_instantiation_in_tree(tree, class_name)


def _check_file(filename: str) -> list[LintMessage]:
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
                LintMessage(
                    path=filename,
                    line=node.lineno,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name=f"[{HW_CLASSIFICATION_ATTR}]",
                    original=None,
                    replacement=None,
                    description=(
                        f"Test class '{node.name}' must declare "
                        f"{HW_CLASSIFICATION_ATTR} = HardwareClassification.<MEMBER>."
                    ),
                )
            )
            continue

        value = _extract_hw_classification_value(assign_node)

        # Value is not HardwareClassification.enum_member
        if value is None:
            messages.append(
                LintMessage(
                    path=filename,
                    line=assign_node.lineno,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name=f"[{HW_CLASSIFICATION_ATTR}]",
                    original=None,
                    replacement=None,
                    description=(
                        f"Could not determine {HW_CLASSIFICATION_ATTR} value for class '{node.name}'. "
                        f"Use 'HardwareClassification.<MEMBER>'."
                    ),
                )
            )
        elif value == ACCELERATOR:
            messages.extend(_check_accelerator_methods(node, filename))
            if not _class_is_instantiated(node.name, rel_path, tree):
                messages.append(
                    LintMessage(
                        path=filename,
                        line=node.lineno,
                        char=None,
                        code=LINTER_CODE,
                        severity=LintSeverity.ERROR,
                        name=f"[{HW_CLASSIFICATION_ATTR}]",
                        original=None,
                        replacement=None,
                        description=(
                            f"ACCELERATOR class '{node.name}' must be "
                            f"instantiated via 'instantiate_device_type_tests'."
                        ),
                    )
                )

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure test classes declare hw_classification.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    with mp.Pool(8) as pool:
        results = pool.map(_check_file, args.filenames)

    for msgs in results:
        for msg in msgs:
            print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
