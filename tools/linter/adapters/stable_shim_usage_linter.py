#!/usr/bin/env python3
"""
STABLE_SHIM_USAGE: Ensures that calls to versioned shim functions from
torch/csrc/stable/c/shim.h in torch/csrc/stable are properly wrapped in
TORCH_FEATURE_VERSION macros corresponding to the version when those
functions were introduced.
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


LINTER_CODE = "STABLE_SHIM_USAGE"


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


def get_shim_functions() -> dict[str, tuple[int, int]]:
    """
    Extract function names from shim.h and their required version.
    Returns a dict mapping function name to (major, minor) version tuple.

    Functions defined inside TORCH_FEATURE_VERSION blocks require that version.
    """
    repo_root = Path(__file__).resolve().parents[3]
    shim_file = repo_root / "torch/csrc/stable/c/shim.h"

    if not shim_file.exists():
        raise RuntimeError(
            f"Could not find shim.h at {shim_file}. "
            f"This linter requires torch/csrc/stable/c/shim.h to exist in the repository."
        )

    functions: dict[str, tuple[int, int]] = {}

    with open(shim_file) as f:
        lines = f.readlines()

    # Track ALL preprocessor conditional blocks to properly match #if/#endif pairs
    # Each element is (is_version_block, version_tuple_or_none)
    preprocessor_stack: list[tuple[bool, tuple[int, int] | None]] = []
    current_version: tuple[int, int] | None = None

    version_pattern = re.compile(
        r"#if\s+TORCH_FEATURE_VERSION\s*>=\s*TORCH_VERSION_(\d+)_(\d+)_\d+"
    )
    # Match function declarations like: AOTI_TORCH_EXPORT ... function_name(
    function_pattern = re.compile(r"AOTI_TORCH_EXPORT\s+\w+\s+(\w+)\s*\(")
    # Also match typedef function pointers
    typedef_pattern = re.compile(r"typedef\s+.*\(\*(\w+)\)")

    for line in lines:
        stripped = line.strip()

        # Skip comments
        if stripped.startswith("//"):
            continue

        # Check for TORCH_FEATURE_VERSION block start
        version_match = version_pattern.match(stripped)
        if version_match:
            major = int(version_match.group(1))
            minor = int(version_match.group(2))
            version_tuple = (major, minor)
            preprocessor_stack.append((True, version_tuple))
            current_version = version_tuple
            continue

        # Track any other #if/#ifdef/#ifndef directives
        if stripped.startswith(("#if", "#ifdef", "#ifndef")) and not version_match:
            preprocessor_stack.append((False, None))
            continue

        # Track #endif directives
        if stripped.startswith("#endif"):
            if preprocessor_stack:
                is_version_block, _ = preprocessor_stack.pop()
                # If we just closed a version block, check if we're still in one
                if is_version_block:
                    current_version = None
                    for is_ver, ver_tuple in reversed(preprocessor_stack):
                        if is_ver:
                            current_version = ver_tuple
                            break
            continue

        # Track #else and #elif (exit version blocks on the topmost block)
        if stripped.startswith(("#else", "#elif")):
            if preprocessor_stack and preprocessor_stack[-1][0]:
                current_version = None
            continue

        # Check for function declarations
        func_match = function_pattern.search(stripped)
        if func_match and current_version:
            func_name = func_match.group(1)
            functions[func_name] = current_version
            continue

        typedef_match = typedef_pattern.search(stripped)
        if typedef_match and current_version:
            func_name = typedef_match.group(1)
            functions[func_name] = current_version
            continue

    return functions


def check_file(filename: str) -> list[LintMessage]:
    """
    Check if the file is in torch/csrc/stable and lint it for proper
    usage of versioned shim functions.
    """
    lint_messages: list[LintMessage] = []

    # Only lint files in torch/csrc/stable (but not the shim itself)
    if "torch/csrc/stable" not in filename or filename.endswith("c/shim.h"):
        return []

    # Get versioned shim functions
    shim_functions = get_shim_functions()
    if not shim_functions:
        raise RuntimeError("Could not extract any shim_functions")

    with open(filename) as f:
        lines = f.readlines()

    # Track current version block
    inside_version_block = False
    current_version_macro: tuple[int, int] | None = None

    # Track ALL preprocessor conditional blocks to properly match #if/#endif pairs
    # Each element is (is_version_block, version_macro_or_none)
    preprocessor_stack: list[tuple[bool, tuple[int, int] | None]] = []

    version_start_pattern = re.compile(
        r"#if\s+TORCH_FEATURE_VERSION\s*>=\s*TORCH_VERSION_(\d+)_(\d+)_\d+"
    )

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip commented out lines - they're not active preprocessor directives
        if stripped.startswith("//"):
            continue

        # Check for TORCH_FEATURE_VERSION block start
        version_start_match = version_start_pattern.search(line)
        if version_start_match:
            major = int(version_start_match.group(1))
            minor = int(version_start_match.group(2))
            version_macro = (major, minor)
            preprocessor_stack.append((True, version_macro))
            inside_version_block = True
            current_version_macro = version_macro
            continue

        # Track any other #if/#ifdef/#ifndef directives
        if stripped.startswith(("#if", "#ifdef", "#ifndef")):
            # Not a TORCH_FEATURE_VERSION block, just a regular conditional
            preprocessor_stack.append((False, None))
            continue

        # Track #endif directives
        if stripped.startswith("#endif"):
            if preprocessor_stack:
                is_version_block, _ = preprocessor_stack.pop()
                # If we just closed a version block, check if we're still in one
                if is_version_block:
                    # Look for any remaining version blocks in the stack
                    inside_version_block = False
                    current_version_macro = None
                    for is_ver, ver_macro in reversed(preprocessor_stack):
                        if is_ver:
                            inside_version_block = True
                            current_version_macro = ver_macro
                            break
            continue

        # Track #else and #elif (they don't change the stack depth, but exit version blocks)
        if stripped.startswith(("#else", "#elif")):
            # If we're in a version block, exit it (the #else branch is not versioned)
            if inside_version_block and preprocessor_stack:
                # Check if the topmost block is a version block
                if preprocessor_stack[-1][0]:
                    inside_version_block = False
                    current_version_macro = None
            continue

        # Check for calls to versioned shim functions
        for func_name, required_version in shim_functions.items():
            # Look for function calls like: func_name(
            # Use word boundaries to avoid matching partial names
            if re.search(rf"\b{re.escape(func_name)}\s*\(", line):
                major, minor = required_version
                required_macro = f"TORCH_VERSION_{major}_{minor}_0"

                if not inside_version_block:
                    lint_messages.append(
                        LintMessage(
                            path=filename,
                            line=line_num,
                            char=None,
                            code=LINTER_CODE,
                            severity=LintSeverity.ERROR,
                            name="unversioned-shim-call",
                            original=None,
                            replacement=None,
                            description=(
                                f"Call to versioned shim function '{func_name}' is not wrapped "
                                f"in a TORCH_FEATURE_VERSION block. This function requires at least:\n"
                                f"#if TORCH_FEATURE_VERSION >= {required_macro}\n"
                                f"  // ... your code calling {func_name} ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {required_macro}"
                            ),
                        )
                    )
                elif (
                    current_version_macro is not None
                    and current_version_macro < required_version
                ):
                    # Error only if current version is LESS than required version
                    # If current version is >= required version, that's fine
                    current_major, current_minor = current_version_macro
                    current_macro = f"TORCH_VERSION_{current_major}_{current_minor}_0"
                    lint_messages.append(
                        LintMessage(
                            path=filename,
                            line=line_num,
                            char=None,
                            code=LINTER_CODE,
                            severity=LintSeverity.ERROR,
                            name="insufficient-version-for-shim-call",
                            original=None,
                            replacement=None,
                            description=(
                                f"Call to '{func_name}' is wrapped in {current_macro}, "
                                f"but this function requires at least {required_macro}. "
                                f"The version guard must be at least the required version:\n"
                                f"#if TORCH_FEATURE_VERSION >= {required_macro}\n"
                                f"  // ... your code calling {func_name} ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {required_macro}"
                            ),
                        )
                    )

    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="stable shim usage linter",
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
