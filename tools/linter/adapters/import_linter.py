"""
Checks files to make sure there are no imports from disallowed third party
libraries.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import token
from enum import Enum
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter


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


LINTER_CODE = "IMPORT_LINTER"
CURRENT_FILE_NAME = os.path.basename(__file__)
_MODULE_NAME_ALLOW_LIST: set[str] = set()

# Add builtin modules of python.
_MODULE_NAME_ALLOW_LIST.update(sys.stdlib_module_names)

# Add the allowed third party libraries. Please avoid updating this unless you
# understand the risks -- see `_ERROR_MESSAGE` for why.
_MODULE_NAME_ALLOW_LIST.update(
    [
        "sympy",
        "einops",
        "libfb",
        "torch",
        "tvm",
        "_pytest",
        "tabulate",
        "optree",
        "typing_extensions",
        "triton",
        "functorch",
        "torchrec",
        "numpy",
        "torch_xla",
        "annotationlib",  # added in python 3.14
    ]
)

_ERROR_MESSAGE = """
Please do not import third-party modules in PyTorch unless they're explicit
requirements of PyTorch. Imports of a third-party library may have side effects
and other unintentional behavior. If you're just checking if a module exists,
use sys.modules.get("torchrec") or the like.
"""

_OPTIONAL_IMPORT_TIME_DENY_LIST = {
    "_pytest",
    "einops",
    "optree",
    "torchrec",
    "triton",
}

_IMPORT_TIME_CALL_DENY_LIST = {
    "has_triton_package",
}

_IMPORT_TIME_ERROR_MESSAGE = """
Please do not import optional third-party modules at torch._dynamo import time.
Imports of optional third-party libraries may have side effects and other
unintentional behavior. If you're checking whether a module has already been
imported, use sys.modules.get("torchrec") or the like.
"""

_IMPORT_TIME_EXCLUDED_PATH_PREFIXES = ("torch/_dynamo/repro/",)


def _module_name_from_import(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.level != 0 or node.module is None:
        return []
    return [node.module.split(".")[0]]


def _is_type_checking_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


class ImportTimeVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.lint_messages: list[LintMessage] = []
        self._function_depth = 0
        self._type_checking_depth = 0

    @property
    def _is_import_time(self) -> bool:
        return self._function_depth == 0 and self._type_checking_depth == 0

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_expr(node.test):
            self._type_checking_depth += 1
            for stmt in node.body:
                self.visit(stmt)
            self._type_checking_depth -= 1
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._function_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._function_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        self._check_import(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check_import(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_call(node)
        self.generic_visit(node)

    def _add_import_time_message(
        self, node: ast.Import | ast.ImportFrom | ast.Call, name: str
    ) -> None:
        self.lint_messages.append(
            LintMessage(
                path=self.filepath,
                line=node.lineno,
                char=node.col_offset,
                code="IMPORT",
                severity=LintSeverity.ERROR,
                name=name,
                original=None,
                replacement=None,
                description=_IMPORT_TIME_ERROR_MESSAGE,
            )
        )

    def _check_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if not self._is_import_time:
            return
        for module_name in _module_name_from_import(node):
            if module_name in _OPTIONAL_IMPORT_TIME_DENY_LIST:
                self._add_import_time_message(node, "Disallowed import-time import")

    def _check_call(self, node: ast.Call) -> None:
        if not self._is_import_time:
            return
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            return
        if name in _IMPORT_TIME_CALL_DENY_LIST:
            self._add_import_time_message(node, "Disallowed import-time call")


def check_import_time_imports(filepath: str) -> list[LintMessage]:
    path = Path(filepath)
    try:
        path = path.resolve().relative_to(_linter.ROOT)
    except ValueError:
        pass
    path_str = str(path).replace(os.sep, "/")
    if path_str.startswith(_IMPORT_TIME_EXCLUDED_PATH_PREFIXES):
        return []
    tree = ast.parse(Path(filepath).read_text(), filename=filepath)
    visitor = ImportTimeVisitor(filepath)
    visitor.visit(tree)
    return visitor.lint_messages


def check_file(filepath: str) -> list[LintMessage]:
    path = Path(filepath)
    file = _linter.PythonFile("import_linter", path=path)
    lint_messages = check_import_time_imports(filepath)
    for line_of_tokens in file.token_lines:
        # Skip indents
        idx = 0
        for tok in line_of_tokens:
            if tok.type == token.INDENT:
                idx += 1
            else:
                break

        # Look for either "import foo..." or "from foo..."
        if idx + 1 < len(line_of_tokens):
            tok0 = line_of_tokens[idx]
            tok1 = line_of_tokens[idx + 1]
            if tok0.type == token.NAME and tok0.string in {"import", "from"}:
                if tok1.type == token.NAME:
                    module_name = tok1.string
                    if module_name not in _MODULE_NAME_ALLOW_LIST:
                        line = tok0.start[0]
                        msg = LintMessage(
                            path=filepath,
                            line=line,
                            char=None,
                            code="IMPORT",
                            severity=LintSeverity.ERROR,
                            name="Disallowed import",
                            original=None,
                            replacement=None,
                            description=_ERROR_MESSAGE,
                        )
                        lint_messages.append(msg)
    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filepaths",
        nargs="+",
        help="paths of files to lint",
    )
    args = parser.parse_args()

    # Check all files.
    all_lint_messages = []
    for filepath in args.filepaths:
        lint_messages = check_file(filepath)
        all_lint_messages.extend(lint_messages)

    # Print out lint messages.
    for lint_message in all_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
