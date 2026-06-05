#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "TEST_HARDWARE_REQUIREMENT"

HARDWARE_REQUIREMENT_ATTR = "hardware_requirement"


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


def is_test_file(path: Path) -> bool:
    return (
        path.name == "test.py"
        or path.name.startswith("test_")
        or path.name.endswith("_test.py")
    )


def get_base_name(base: ast.expr) -> str | None:
    # class XXX(TestCase)
    # class XXX(YYYTestCase)
    if isinstance(base, ast.Name):
        return base.id
    # class XXX(YYY.TestCase)
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def is_test_case_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_name = get_base_name(base)
        if base_name is not None and base_name.endswith("TestCase"):
            return True
    return False


def has_hardware_requirement(node: ast.ClassDef) -> bool:
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == HARDWARE_REQUIREMENT_ATTR
                ):
                    return True
        if isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if isinstance(target, ast.Name) and target.id == HARDWARE_REQUIREMENT_ATTR:
                return True
    return False


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename)
    if not is_test_file(path):
        return []

    with open(path) as f:
        tree = ast.parse(f.read(), filename=filename)

    lint_messages = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not is_test_case_class(node):
            continue
        if has_hardware_requirement(node):
            continue

        lint_messages.append(
            LintMessage(
                path=filename,
                line=node.lineno,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name=f"[{HARDWARE_REQUIREMENT_ATTR}]",
                original=None,
                replacement=None,
                description=(
                    f"Test class '{node.name}' must define a class-level "
                    f"{HARDWARE_REQUIREMENT_ATTR} attribute."
                ),
            )
        )

    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure selected test suites declare hardware_requirement.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    with mp.Pool(8) as pool:
        lint_messages = pool.map(check_file, args.filenames)

    for lint_message in [item for sublist in lint_messages for item in sublist]:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
