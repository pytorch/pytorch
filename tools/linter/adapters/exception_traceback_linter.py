"""
Linter that checks for exception reference cycles.

When an exception caught with `except ... as e:` is stored beyond the except
block (e.g. `saved = e`), its `__traceback__` attribute keeps the entire call
stack alive, creating a reference cycle.  This linter flags such patterns and
suggests adding `traceback.clear_frames(e.__traceback__)` to break the cycle.
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
    "Add `traceback.clear_frames({exc_name}.__traceback__)` or "
    "`{exc_name}.__traceback__ = None` to avoid reference cycles. "
    "See https://docs.python.org/3/reference/compound_stmts.html#the-try-statement"
)


class _TracebackVisitor(ast.NodeVisitor):
    """Walk an except-handler body looking for assignments that leak the
    exception variable and for traceback cleanup patterns."""

    def __init__(self, exc_name: str) -> None:
        self.exc_name = exc_name
        # Names that alias the exception (including the original)
        self.aliases: set[str] = {exc_name}
        self.is_stored = False
        self.is_cleared = False
        self.is_reraised = False
        self.store_line: int | None = None

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track aliases: `exc = e` means `exc` is also an alias
        if self._value_is_exc(node.value):
            # Only flag simple name targets (e.g. `saved = e`).
            # Attribute assignments like `ret.__cause__ = e` are standard
            # exception chaining and not the kind of leak we're looking for.
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases.add(target.id)
                    self.is_stored = True
                    if self.store_line is None:
                        self.store_line = node.lineno
        self._check_clear(node)
        self.generic_visit(node)

    def _check_clear(self, node: ast.Assign) -> None:
        """Detect `alias.__traceback__ = None`."""
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "__traceback__"
                and isinstance(target.value, ast.Name)
                and target.value.id in self.aliases
            ):
                self.is_cleared = True

    def visit_Call(self, node: ast.Call) -> None:
        """Detect `traceback.clear_frames(alias.__traceback__)`."""
        if self._is_clear_frames_call(node) and node.args:
            arg = node.args[0]
            if (
                isinstance(arg, ast.Attribute)
                and arg.attr == "__traceback__"
                and isinstance(arg.value, ast.Name)
                and arg.value.id in self.aliases
            ):
                self.is_cleared = True
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        """Bare `raise` or `raise alias` means the exception escapes via
        normal raise semantics -- not a leak from storage."""
        if node.exc is None:
            # bare `raise`
            self.is_reraised = True
        elif isinstance(node.exc, ast.Name) and node.exc.id in self.aliases:
            self.is_reraised = True
        self.generic_visit(node)

    # Don't descend into nested scopes -- a bare `raise` or `clear_frames`
    # inside a nested except/function/lambda does not apply to the outer
    # exception variable.
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def _value_is_exc(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id in self.aliases

    @staticmethod
    def _is_clear_frames_call(node: ast.Call) -> bool:
        # traceback.clear_frames(...)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "clear_frames"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "traceback"
        ):
            return True
        # from traceback import clear_frames; clear_frames(...)
        if isinstance(node.func, ast.Name) and node.func.id == "clear_frames":
            return True
        return False


class _FileChecker(ast.NodeVisitor):
    def __init__(self, filepath: str, source_lines: list[str]) -> None:
        self.filepath = filepath
        self.source_lines = source_lines
        self.messages: list[LintMessage] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            visitor = _TracebackVisitor(node.name)
            for child in ast.iter_child_nodes(node):
                visitor.visit(child)
            if (
                visitor.is_stored
                and not visitor.is_cleared
                and not visitor.is_reraised
                and not self._has_noqa(visitor.store_line)
            ):
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

    def _has_noqa(self, lineno: int | None) -> bool:
        if lineno is None:
            return False
        line = self.source_lines[lineno - 1]
        noqa_tag = "#" + " noqa"
        if noqa_tag not in line:
            return False
        idx = line.index(noqa_tag)
        rest = line[idx + len(noqa_tag) :]
        if not rest.strip() or rest.strip().startswith(":"):
            if not rest.strip():
                return True
            codes = rest.strip()[1:]
            return LINTER_CODE in codes
        return False


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
        return []
    source_lines = source.splitlines()
    checker = _FileChecker(filepath, source_lines)
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
