#!/usr/bin/env python3
"""
This lint verifies that tests under test/ should use ``torch.library._scoped_library`` instead
of ``torch.library.Library()``. It replaces the old ``TOR901`` check from removed torchfix.

Skip linting with ``# noqa: SCOPED_LIBRARY`` on the line or above
"""

from __future__ import annotations

import argparse
import ast
import json
from enum import Enum
from typing import NamedTuple


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


LINTER_CODE = "SCOPED_LIBRARY"


def _is_noqa_suppressed(source_lines: list[str], lineno: int) -> bool:
    if lineno <= 0 or lineno > len(source_lines):
        return False
    if source_lines[lineno - 1].rstrip().endswith(f"# noqa: {LINTER_CODE}"):
        return True
    if lineno > 1 and source_lines[lineno - 2].strip() == f"# noqa: {LINTER_CODE}":
        return True
    return False


def _is_torch_library_call(node: ast.AST) -> bool:
    """True if the call is likely torch.library Library construction."""
    if isinstance(node, ast.Name) and node.id == "Library":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Library":
        s = ast.unparse(node)
        return (
            s
            in (
                "torch.library.Library",
                "library.Library",
            )
            or s.endswith(".Library")
            and "library" in s
        )
    return False


def _call_suppressed(source_lines: list[str], node: ast.Call) -> bool:
    """True if any line of the call is covered by noqa."""
    start = node.lineno
    end = getattr(node, "end_lineno", None) or start
    for line_no in range(start, end + 1):
        if _is_noqa_suppressed(source_lines, line_no):
            return True
    return False


def check_file(path: str) -> list[LintMessage]:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError:
        return []

    lines = content.splitlines()
    lint_messages = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_torch_library_call(node.func):
            continue
        line_no = node.lineno
        if _call_suppressed(lines, node):
            continue
        lint_messages.append(
            LintMessage(
                path=path,
                line=line_no,
                char=node.col_offset,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="direct-torch-Library",
                original=None,
                replacement=None,
                description=(
                    f"In tests, use torch.library._scoped_library, or skip linting with '# noqa: {LINTER_CODE}'"
                ),
            )
        )
    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enforce _scoped_library in the PyTorch test tree",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="Python files to check",
    )
    args = parser.parse_args()
    for path in args.filenames:
        for m in check_file(path):
            print(json.dumps(m._asdict()), flush=True)


if __name__ == "__main__":
    main()
