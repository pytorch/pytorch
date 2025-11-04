#!/usr/bin/env python3
"""
STABLE_NATIVE_FUNCTION_SCHEMA: Ensures that when a native function schema is changed
in native_functions.yaml for an op that is used in torch/csrc/stable/ops.h,
a schema adapter is registered in torch/csrc/shim_common.cpp.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "STABLE_NATIVE_FUNCTION_SCHEMA"


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


def read_native_ops_txt(native_ops_txt_path: Path) -> set[str]:
    """
    Read the list of ops from native_ops.txt that we care about.

    Each line should contain one op name.
    Empty lines and lines starting with # are ignored.
    """
    ops = set()

    if not native_ops_txt_path.exists():
        raise RuntimeError("Could not find torch/csrc/stable/native_ops.txt")

    with open(native_ops_txt_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            ops.add(line)

    return ops


def get_changed_function_schemas(
    native_functions_yaml_path: Path,
) -> dict[str, tuple[str, str]]:
    """
    Get the function schemas that were changed in the most recent commit or uncommitted changes.

    Returns a dict mapping op name to (old_schema, new_schema) tuples.
    """
    changed_schemas = {}

    # Check uncommitted changes (working directory vs HEAD)
    result = subprocess.run(
        ["git", "diff", "HEAD", str(native_functions_yaml_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    diff_output = result.stdout

    # Also check the most recent commit (HEAD vs HEAD~1)
    result = subprocess.run(
        ["git", "diff", "HEAD~1..HEAD", str(native_functions_yaml_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    diff_output += "\n" + result.stdout

    # Parse the diff to find changed function schemas
    # We're looking for lines like:
    # - func: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
    # + func: transpose.int(Tensor(a) self, int dim0, int dim1, *, int new_arg=0) -> Tensor(a)

    current_op: str | None = None
    old_schema: str | None = None
    new_schema: str | None = None

    for line in diff_output.split("\n"):
        # Check for removed schema line (old version)
        # Note: In YAML diffs, lines like "- func:" appear as "-- func:" in git diff
        if line.startswith("-") and "func:" in line:
            # Match op name: everything after "func:" up to the opening paren
            match = re.search(r"func:\s*([^\s(]+)", line)
            if match:
                current_op = match.group(1)
                # Extract the full schema (skip the leading "- ")
                schema_match = re.search(r"func:\s*(.+)", line[1:].lstrip())
                if schema_match:
                    old_schema = schema_match.group(1).strip()

        # Check for added schema line (new version)
        # Note: In YAML diffs, lines like "- func:" appear as "+- func:" in git diff
        elif line.startswith("+") and "func:" in line and current_op:
            # Match op name: everything after "func:" up to the opening paren
            match = re.search(r"func:\s*([^\s(]+)", line)
            if match and match.group(1) == current_op:
                # Extract the full schema (skip the leading "+ ")
                schema_match = re.search(r"func:\s*(.+)", line[1:].lstrip())
                if schema_match:
                    new_schema = schema_match.group(1).strip()

                    # Store the change (old_schema must exist since current_op is set)
                    if old_schema:
                        changed_schemas[current_op] = (old_schema, new_schema)

                        # Reset for next op
                        current_op = None
                        old_schema = None
                        new_schema = None

    return changed_schemas


def get_registered_adapters(shim_common_path: Path) -> set[str]:
    """
    Get the list of ops that have schema adapters registered in shim_common.cpp.

    Looks for calls to register_schema_adapter("aten::op_name", ...).
    """
    adapters = set()

    with open(shim_common_path) as f:
        content = f.read()

    # Pattern to match register_schema_adapter("aten::...", ...)
    pattern = re.compile(r'register_schema_adapter\s*\(\s*"(aten::[^"]+)"')

    for match in pattern.finditer(content):
        op_name = match.group(1)
        adapters.add(op_name)

    return adapters


def check_file(filename: str) -> list[LintMessage]:
    """
    Check if native_functions.yaml has schema changes for ops in native_ops.txt
    that don't have adapters registered in shim_common.cpp.
    """
    lint_messages: list[LintMessage] = []

    # Only lint aten/src/ATen/native/native_functions.yaml
    if not filename.endswith("native/native_functions.yaml"):
        return []

    repo_root = Path(__file__).resolve().parents[3]
    native_ops_txt_path = repo_root / "torch/csrc/stable/native_ops.txt"
    shim_common_path = repo_root / "torch/csrc/shim_common.cpp"
    native_functions_yaml_path = Path(filename)

    # Get the list of ops we care about
    tracked_ops = read_native_ops_txt(native_ops_txt_path)

    changed_schemas = get_changed_function_schemas(native_functions_yaml_path)

    if not changed_schemas:
        # No schema changes detected
        return lint_messages

    # Get registered adapters
    registered_adapters = get_registered_adapters(shim_common_path)

    # Check if any tracked ops have schema changes without adapters
    for op_name, (old_schema, new_schema) in changed_schemas.items():
        # Check if this op is in our tracked list
        # Handle both "aten::op" and "aten::op.overload" formats
        base_op = f"aten::{op_name.split('.')[0]}"
        full_op = f"aten::{op_name}"

        is_tracked = any(
            tracked_op == full_op or tracked_op.startswith(base_op)
            for tracked_op in tracked_ops
        )

        if is_tracked:
            # Check if an adapter is registered
            has_adapter = (
                full_op in registered_adapters or base_op in registered_adapters
            )

            if not has_adapter:
                lint_messages.append(
                    LintMessage(
                        path=filename,
                        line=None,
                        char=None,
                        code=LINTER_CODE,
                        severity=LintSeverity.ERROR,
                        name="missing-schema-adapter",
                        original=None,
                        replacement=None,
                        description=(
                            f"Schema change detected for '{op_name}' which is used in torch/csrc/stable/ops.h:\n\n"
                            f"Old schema:\n  {old_schema}\n\n"
                            f"New schema:\n  {new_schema}\n\n"
                            f"Please register a schema adapter in torch/csrc/shim_common.cpp\n"
                            f"in the _register_adapters() function to handle this schema change.\n"
                            f"See https://github.com/pytorch/pytorch/pull/165284/ for an example."
                        ),
                    )
                )

    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native function schema linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    lint_messages = []
    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
