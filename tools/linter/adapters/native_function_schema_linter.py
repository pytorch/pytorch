#!/usr/bin/env python3
"""
STABLE_NATIVE_FUNCTION_SCHEMA: Ensures that:
1. Auto-generates native_ops.txt from ops.h (like shim_function_versions.txt)
2. When a native function schema is changed in native_functions.yaml for an op
   that is used in torch/csrc/stable/ops.h, a schema adapter is registered in
   torch/csrc/shim_common.cpp.
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

from torchgen.model import FunctionSchema


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


def extract_native_ops_from_ops_h(ops_h_path: Path) -> set[str]:
    """
    Extract all ATen op names called via torch_call_dispatcher in ops.h.
    Returns a set of op names like "aten::empty_like", "aten::transpose.int", etc.
    """
    ops = set()

    with open(ops_h_path) as f:
        content = f.read()

    pattern = re.compile(
        r'torch_call_dispatcher\s*\(\s*"(aten::[^"]+)"\s*,\s*"([^"]*)"\s*,'
    )

    for match in pattern.finditer(content):
        op_name = match.group(1)
        overload = match.group(2)

        if overload:
            full_op_name = f"{op_name}.{overload}"
        else:
            full_op_name = op_name

        ops.add(full_op_name)

    return ops


def write_native_ops_txt(ops: set[str], output_file: Path) -> None:
    """
    Write the native ops list to a text file.
    """
    sorted_ops = sorted(ops)

    with open(output_file, "w") as f:
        f.write(
            "# Auto-generated file listing ATen native ops used in torch/csrc/stable/ops.h\n"
        )
        f.write(
            "# Each line contains one op name in format: aten::op_name or aten::op_name.overload\n"
        )
        f.write("#\n")
        f.write(
            "# This file is automatically updated by the stable_native_function_schema_linter.\n"
        )
        f.write("# DO NOT EDIT MANUALLY.\n\n")

        for op in sorted_ops:
            f.write(f"{op}\n")


def read_native_ops_txt(native_ops_txt_path: Path) -> set[str]:
    """
    Read the list of ops from native_ops.txt.
    """
    ops = set()

    if not native_ops_txt_path.exists():
        raise RuntimeError("Could not find torch/csrc/stable/native_ops.txt")

    with open(native_ops_txt_path) as f:
        for line in f:
            line = line.strip()
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


def get_registered_adapters(shim_common_path: Path) -> dict[str, tuple[int, int]]:
    """
    Get the list of ops that have schema adapters registered in shim_common.cpp.

    Looks for calls to register_schema_adapter("aten::op_name", TORCH_VERSION_X_Y_Z, ...).
    Returns a dict mapping op name to (major, minor) version tuple.
    """
    adapters = {}

    with open(shim_common_path) as f:
        content = f.read()

    # Pattern to match register_schema_adapter("aten::...", TORCH_VERSION_X_Y_Z, ...)
    pattern = re.compile(
        r'register_schema_adapter\s*\(\s*"(aten::[^"]+)"\s*,\s*TORCH_VERSION_(\d+)_(\d+)_\d+'
    )

    for match in pattern.finditer(content):
        op_name = match.group(1)
        major = int(match.group(2))
        minor = int(match.group(3))
        adapters[op_name] = (major, minor)

    return adapters


def get_current_torch_version() -> tuple[int, int]:
    """
    Get the current PyTorch major and minor version from version.txt.
    Returns (major, minor) tuple.
    """
    repo_root = Path(__file__).resolve().parents[3]
    version_file = repo_root / "version.txt"

    with open(version_file) as f:
        version_str = f.read().strip()

    # Parse version like "2.10.0a0" -> (2, 10)
    match = re.match(r"(\d+)\.(\d+)\.", version_str)
    if not match:
        raise RuntimeError(f"Could not parse version from {version_str}")

    return (int(match.group(1)), int(match.group(2)))


def check_return_types_match(
    old_func: FunctionSchema, new_func: FunctionSchema
) -> None:
    """
    Check if return types match between two function schemas.

    Raises RuntimeError if they don't match, as this indicates an unusual case
    that should be reported to the PyTorch team.
    """
    if len(old_func.returns) != len(new_func.returns):
        raise RuntimeError(
            f"Return type mismatch for '{old_func.name}': number of returns changed "
            f"from {len(old_func.returns)} to {len(new_func.returns)}. "
            f"This is unexpected. Please file an issue at "
            f"https://github.com/pytorch/pytorch/issues"
        )

    for old_ret, new_ret in zip(old_func.returns, new_func.returns):
        if old_ret.type != new_ret.type:
            raise RuntimeError(
                f"Return type mismatch for '{old_func.name}': type changed "
                f"from {old_ret.type} to {new_ret.type}. "
                f"This is unexpected. Please file an issue at "
                f"https://github.com/pytorch/pytorch/issues"
            )
        if old_ret.annotation != new_ret.annotation:
            raise RuntimeError(
                f"Return type annotation mismatch for '{old_func.name}': "
                f"annotation changed from {old_ret.annotation} to {new_ret.annotation}. "
                f"This is unexpected. Please file an issue at "
                f"https://github.com/pytorch/pytorch/issues"
            )


def is_only_default_arg_value_change(
    old_schema: str, new_schema: str
) -> tuple[bool, list[str]]:
    """
    Determine if the schema change is ONLY a default argument value change.
    Parameter name changes are also allowed (they don't affect ABI).

    Returns (is_default_only, changed_args) where:
    - is_default_only: True if only default values changed (or param names)
    - changed_args: list of argument positions whose defaults changed
    """
    old_func = FunctionSchema.parse(old_schema)
    new_func = FunctionSchema.parse(new_schema)

    if old_func.name != new_func.name:
        return (False, [])

    # Check return types - raises RuntimeError if they don't match
    check_return_types_match(old_func, new_func)

    old_args = list(old_func.schema_order_arguments())
    new_args = list(new_func.schema_order_arguments())

    if len(old_args) != len(new_args):
        return (False, [])

    changed_args = []

    for i, (old_arg, new_arg) in enumerate(zip(old_args, new_args)):
        if old_arg.type != new_arg.type:
            return (False, [])

        if old_arg.default != new_arg.default:
            changed_args.append(new_arg.name)

    return (True, changed_args)


def check_file(
    filename: str,
    native_functions_yaml_path: Path,
    shim_common_path: Path,
) -> list[LintMessage]:
    """
    Check if native_functions.yaml has schema changes for ops in native_ops.txt
    that don't have adapters registered in shim_common.cpp.
    """
    lint_messages: list[LintMessage] = []

    repo_root = Path(__file__).resolve().parents[3]
    native_ops_txt_path = repo_root / "torch/csrc/stable/native_ops.txt"

    tracked_ops = read_native_ops_txt(native_ops_txt_path)

    changed_schemas = get_changed_function_schemas(native_functions_yaml_path)

    if not changed_schemas:
        # No schema changes detected
        return lint_messages

    # Get registered adapters
    registered_adapters = get_registered_adapters(shim_common_path)

    # Get current PyTorch version
    current_version = get_current_torch_version()

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
            adapter_key = None
            if full_op in registered_adapters:
                adapter_key = full_op
            elif base_op in registered_adapters:
                adapter_key = base_op

            if not adapter_key:
                correct_version = (
                    f"TORCH_VERSION_{current_version[0]}_{current_version[1]}_0"
                )

                # Determine if this is only a default value change
                only_default_change, changed_args = is_only_default_arg_value_change(
                    old_schema, new_schema
                )

                if only_default_change:
                    # WARNING: Can use version guards in ops.h
                    args_str = ", ".join(changed_args) if changed_args else "argument"
                    lint_messages.append(
                        LintMessage(
                            path=filename,
                            line=None,
                            char=None,
                            code=LINTER_CODE,
                            severity=LintSeverity.WARNING,
                            name="default-value-change",
                            original=None,
                            replacement=None,
                            description=(
                                f"Default value changed for {args_str} in '{op_name}': "
                                f"update the default value in torch/csrc/stable/ops.h with version guards "
                                f"(#if TORCH_FEATURE_VERSION >= {correct_version})."
                            ),
                        )
                    )
                else:
                    # ERROR: Structural change requires adapter
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
                                f"Structural schema change detected for '{op_name}' which is used in torch/csrc/stable/ops.h:\n\n"
                                f"Old schema:\n  {old_schema}\n\n"
                                f"New schema:\n  {new_schema}\n\n"
                                f"This change requires a schema adapter because it's not just a default value change.\n"
                                f"Please register a schema adapter in torch/csrc/shim_common.cpp\n"
                                f"in the _register_adapters() function with {correct_version}.\n"
                                f"See https://github.com/pytorch/pytorch/pull/165284/ for an example."
                            ),
                        )
                    )
            else:
                # Adapter exists, check if version is correct
                adapter_version = registered_adapters[adapter_key]
                if adapter_version != current_version:
                    lint_messages.append(
                        LintMessage(
                            path=filename,
                            line=None,
                            char=None,
                            code=LINTER_CODE,
                            severity=LintSeverity.ERROR,
                            name="incorrect-adapter-version",
                            original=None,
                            replacement=None,
                            description=(
                                f"Schema adapter for '{op_name}' is registered with incorrect version.\n\n"
                                f"Adapter registered with: TORCH_VERSION_{adapter_version[0]}_{adapter_version[1]}_0\n"
                                f"Current version: TORCH_VERSION_{current_version[0]}_{current_version[1]}_0\n\n"
                                f"Schema change:\n"
                                f"Old schema:\n  {old_schema}\n\n"
                                f"New schema:\n  {new_schema}\n\n"
                                f"Please update the adapter registration in torch/csrc/shim_common.cpp\n"
                                f"to use TORCH_VERSION_{current_version[0]}_{current_version[1]}_0."
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

    repo_root = Path(__file__).resolve().parents[3]
    ops_h_path = repo_root / "torch/csrc/stable/ops.h"
    native_ops_txt_path = repo_root / "torch/csrc/stable/native_ops.txt"
    native_functions_yaml_path = (
        repo_root / "aten/src/ATen/native/native_functions.yaml"
    )
    shim_common_path = repo_root / "torch/csrc/shim_common.cpp"

    ops = extract_native_ops_from_ops_h(ops_h_path)
    write_native_ops_txt(ops, native_ops_txt_path)

    lint_messages = []
    for filename in args.filenames:
        if filename.endswith("native/native_functions.yaml"):
            lint_messages.extend(
                check_file(filename, native_functions_yaml_path, shim_common_path)
            )

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
