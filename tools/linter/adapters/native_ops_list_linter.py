#!/usr/bin/env python3
"""
STABLE_NATIVE_OPS_LIST: Ensures that all ATen ops called via torch_call_dispatcher
in torch/csrc/stable/ops.h are documented in torch/csrc/stable/native_ops.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "STABLE_NATIVE_OPS_LIST"


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
    Extract all ATen op names that are called via torch_call_dispatcher in ops.h.

    Returns a set of op names like "aten::empty_like", "aten::transpose.int", etc.
    """
    ops = set()

    with open(ops_h_path) as f:
        content = f.read()

    # Pattern to match torch_call_dispatcher("aten::...", "overload", ...)
    pattern = re.compile(
        r'torch_call_dispatcher\s*\(\s*"(aten::[^"]+)"\s*,\s*"([^"]*)"\s*,'
    )

    for match in pattern.finditer(content):
        op_name = match.group(1)  # e.g., "aten::empty_like"
        overload = match.group(2)  # e.g., "" or "int"

        # Construct full op name with overload if present
        if overload:
            full_op_name = f"{op_name}.{overload}"
        else:
            full_op_name = op_name

        ops.add(full_op_name)

    return ops


def read_native_ops_txt(native_ops_txt_path: Path) -> set[str]:
    """
    Read the list of documented native ops from native_ops.txt.

    Each line should contain one op name.
    Empty lines and lines starting with # are ignored.
    """
    if not native_ops_txt_path.exists():
        raise RuntimeError(
            f"Could not find native_ops.txt at {native_ops_txt_path}. "
            f"This linter requires torch/csrc/stable/native_ops.txt to exist in the repository."
        )

    ops = set()
    with open(native_ops_txt_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            ops.add(line)

    return ops


def check_file(filename: str) -> list[LintMessage]:
    """
    Check if ops.h has any ops that are not documented in native_ops.txt.
    This check runs when either ops.h or native_ops.txt is modified.
    """
    lint_messages: list[LintMessage] = []

    # Only lint torch/csrc/stable/ops.h or native_ops.txt
    if not (
        filename.endswith(
            ("torch/csrc/stable/ops.h", "torch/csrc/stable/native_ops.txt")
        )
    ):
        return []

    repo_root = Path(__file__).resolve().parents[3]
    ops_h_path = repo_root / "torch/csrc/stable/ops.h"
    native_ops_txt_path = repo_root / "torch/csrc/stable/native_ops.txt"

    # Extract ops from ops.h
    ops_in_code = extract_native_ops_from_ops_h(ops_h_path)

    # Read documented ops from native_ops.txt
    documented_ops = read_native_ops_txt(native_ops_txt_path)

    # Find ops that are in code but not documented
    undocumented_ops = ops_in_code - documented_ops

    if undocumented_ops:
        sorted_ops = sorted(undocumented_ops)
        lint_messages.append(
            LintMessage(
                path=str(ops_h_path),
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="missing-native-ops",
                original=None,
                replacement=None,
                description=(
                    f"The following ATen ops are called in ops.h but not documented in native_ops.txt:\n"
                    f"{chr(10).join('  - ' + op for op in sorted_ops)}\n\n"
                    f"Please add these ops to torch/csrc/stable/native_ops.txt"
                ),
            )
        )

    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native ops list linter",
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
