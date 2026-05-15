"""
Linter that checks for exception reference cycles.

When an exception caught with `except ... as e:` is stored beyond the except
block (e.g. `saved = e`), its `__traceback__` attribute keeps the entire call
stack alive, creating a reference cycle.  This linter flags such patterns and
suggests adding `e.__traceback__ = None` to break the cycle.
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


LINTER_CODE = "EXCEPTION_TRACEBACK"
ERROR_NAME = "exception-traceback-reference-cycle"
DESCRIPTION = (
    "Exception stored outside except block without clearing __traceback__. "
    "Add `traceback.clear_frames({exc_name}.__traceback__)` to avoid reference cycles. "
    "See https://docs.python.org/3/reference/compound_stmts.html#the-try-statement"
)


class _TracebackVisitor(ast.NodeVisitor):
    """Walk an except-handler body looking for assignments that leak the
    exception variable and for ``__traceback__ = None`` clean-ups."""

    def __init__(self, exc_name: str) -> None:
        self.exc_name = exc_name
        self.is_stored = False
        self.is_cleared = False
        self.store_line: int | None = None

    # --- detect `some_var = e` or `obj.attr = e` -------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        if self._value_is_exc(node.value):
            self.is_stored = True
            if self.store_line is None:
                self.store_line = node.lineno
        self._check_clear(node)
        self.generic_visit(node)

    # --- detect `e.__traceback__ = None` or `traceback.clear_frames(e.__traceback__)` ---
    def _check_clear(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "__traceback__"
                and isinstance(target.value, ast.Name)
                and target.value.id == self.exc_name
            ):
                self.is_cleared = True

    def visit_Expr(self, node: ast.Expr) -> None:
        # detect `traceback.clear_frames(e.__traceback__)`
        call = node.value
        if not isinstance(call, ast.Call):
            return
        fn = call.func
        if not (
            isinstance(fn, ast.Attribute)
            and fn.attr == "clear_frames"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "traceback"
        ):
            return
        if len(call.args) == 1:
            arg = call.args[0]
            if (
                isinstance(arg, ast.Attribute)
                and arg.attr == "__traceback__"
                and isinstance(arg.value, ast.Name)
                and arg.value.id == self.exc_name
            ):
                self.is_cleared = True

    # --- detect `raise` (re-raise means no leak) --------------------------
    def visit_Raise(self, node: ast.Raise) -> None:
        # bare `raise` or `raise e` — not a leak
        pass

    # --- helpers ----------------------------------------------------------
    def _value_is_exc(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id == self.exc_name


class _FileChecker(ast.NodeVisitor):
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.messages: list[LintMessage] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            visitor = _TracebackVisitor(node.name)
            for child in ast.iter_child_nodes(node):
                visitor.visit(child)
            if visitor.is_stored and not visitor.is_cleared:
                self.messages.append(
                    LintMessage(
                        path=self.filepath,
                        line=visitor.store_line,
                        char=None,
                        code=LINTER_CODE,
                        severity=LintSeverity.WARNING,
                        name=ERROR_NAME,
                        original=None,
                        replacement=None,
                        description=DESCRIPTION.format(exc_name=node.name),
                    )
                )
        self.generic_visit(node)


def check_file(filepath: str) -> list[LintMessage]:
    try:
        with open(filepath) as f:
            source = f.read()
    except OSError as e:
        return [
            LintMessage(
                path=filepath,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="file-read-error",
                original=None,
                replacement=None,
                description=str(e),
            )
        ]
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        # Not valid Python — skip quietly
        return []
    checker = _FileChecker(filepath)
    checker.visit(tree)
    return checker.messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lint for exception __traceback__ reference cycles.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    for filepath in args.filenames:
        for msg in check_file(filepath):
            print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
