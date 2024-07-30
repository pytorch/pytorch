# mypy: ignore-errors
"""
Lint for uses of set() and set comprehensions in the codebase.
This linter helps identify potential issues with nondeterministic set iteration order.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import traceback
from enum import Enum
from typing import List, NamedTuple


LINTER_CODE = "SETUSAGE"
IGNORE_COMMENT = "# noqa: SETUSAGE"

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


class SetUsageVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, file_content: str):
        self.file_path = file_path
        self.file_content = file_content.splitlines()
        self.issues: List[LintMessage] = []

    def add_issue(self, node: ast.AST, name: str, description: str):
        line = self.file_content[node.lineno - 1]
        if IGNORE_COMMENT not in line:
            self.issues.append(
                LintMessage(
                    path=self.file_path,
                    line=node.lineno,
                    char=node.col_offset,
                    code=LINTER_CODE,
                    severity=LintSeverity.WARNING,
                    name=name,
                    original=None,
                    replacement=None,
                    description=description,
                )
            )

    def visit_Set(self, node: ast.Set):
        self.add_issue(
            node,
            "set-literal",
            "Set literal found. Use OrderedSet for deterministic iteration order.",
        )
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        self.add_issue(
            node,
            "set-comprehension",
            "Set comprehension found. Use OrderedSet for deterministic iteration order.",
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "set":
            self.add_issue(
                node,
                "set-function",
                "set() function call found. Use OrderedSet for deterministic iteration order.",
            )
        self.generic_visit(node)


def check_files(files: List[str], exclude_files: List[str]) -> List[LintMessage]:
    all_issues: List[LintMessage] = []

    for file_path in files:
        if file_path in exclude_files or not file_path.endswith(".py"):
            continue

        try:
            with open(file_path, encoding="utf-8") as file:
                content = file.read()

            tree = ast.parse(content, filename=file_path)
            visitor = SetUsageVisitor(file_path, content)
            visitor.visit(tree)

            all_issues.extend(visitor.issues)
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}\n"
            error_msg += traceback.format_exc()
            logging.error(error_msg)
            all_issues.append(
                LintMessage(
                    path=file_path,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="file-processing-error",
                    original=None,
                    replacement=None,
                    description=error_msg,
                )
            )

    return all_issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set usage linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--filenames",
        required=True,
        help="File containing list of filenames to lint",
    )
    parser.add_argument(
        "--exclude",
        help="File containing list of filenames to exclude from linting",
    )
    args = parser.parse_args()

    with open(args.filenames) as f:
        files_to_check = [line.strip() for line in f if line.strip()]

    exclude_files = []
    if args.exclude:
        with open(args.exclude) as f:
            exclude_files = [line.strip() for line in f if line.strip()]

    lint_messages = check_files(files_to_check, exclude_files)
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)

    sys.exit(0)
