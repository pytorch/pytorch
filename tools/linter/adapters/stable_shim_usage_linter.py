#!/usr/bin/env python3
"""
STABLE_SHIM_USAGE: Ensures that calls to versioned shim functions in
torch/csrc/stable are properly wrapped in TORCH_FEATURE_VERSION macros
corresponding to the version when those functions were introduced.
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


def parse_version(version: str) -> tuple[int, int, int]:
    """
    Parses a version string into (major, minor, patch) version numbers.
    This function is copied from tools/setup_helpers/gen_version_header.py
    to ensure consistency with how PyTorch parses its version.

    Args:
        version: Full version number string, possibly including revision / commit hash.

    Returns:
        A tuple of (major, minor, patch) version numbers.
    """
    # Extract version number part (i.e. toss any revision / hash parts).
    version_number_str = version
    for i in range(len(version)):
        c = version[i]
        if not (c.isdigit() or c == "."):
            version_number_str = version[:i]
            break

    return tuple([int(n) for n in version_number_str.split(".")])  # type: ignore[return-value]


def get_current_version() -> tuple[int, int] | None:
    """
    Get the current PyTorch version from version.txt.
    This uses the same logic as tools/setup_helpers/gen_version_header.py
    which is used to generate torch/headeronly/version.h from version.h.in.

    Returns (major, minor) tuple or None if not found.
    """
    repo_root = Path(__file__).resolve().parents[3]
    version_file = repo_root / "version.txt"

    if not version_file.exists():
        return None

    try:
        with open(version_file) as f:
            version = f.read().strip()
            major, minor, patch = parse_version(version)
            return (major, minor)
    except Exception:
        pass

    return None


def get_added_lines(filename: str) -> set[int]:
    """
    Get the line numbers of added lines in the current uncommitted changes.
    Uses hg/git to determine which lines are new in this commit.

    Returns:
        Set of line numbers (1-indexed) that are new additions.
    """
    import subprocess

    added_lines = set()

    try:
        # Try hg first (Meta's version control)
        result = subprocess.run(
            ["hg", "diff", filename],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            diff_output = result.stdout
        else:
            # Try git as fallback
            result = subprocess.run(
                ["git", "diff", "HEAD", filename],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return added_lines
            diff_output = result.stdout

        # Parse diff output to find added lines
        current_line = 0
        for line in diff_output.split("\n"):
            # Unified diff format: @@ -old_start,old_count +new_start,new_count @@
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                # This is an added line
                added_lines.add(current_line)
                current_line += 1
            elif not line.startswith("-"):
                # Context line or unchanged line
                current_line += 1

    except Exception:
        # If we can't get diff info, return empty set
        pass

    return added_lines


def get_shim_functions() -> dict[str, tuple[int, int]]:
    """
    Extract function names from shim.h and their required version.
    Returns a dict mapping function name to (major, minor) version tuple.

    Functions defined inside TORCH_FEATURE_VERSION blocks require that version.
    """
    repo_root = Path(__file__).resolve().parents[3]
    shim_file = repo_root / "torch/csrc/stable/c/shim.h"

    if not shim_file.exists():
        return {}

    functions: dict[str, tuple[int, int]] = {}

    try:
        with open(shim_file) as f:
            lines = f.readlines()

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

            # Check for version block start
            version_match = version_pattern.match(stripped)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                current_version = (major, minor)
                continue

            # Check for version block end
            if stripped.startswith("#endif") and "TORCH_FEATURE_VERSION" in stripped:
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

    except Exception:
        pass

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

    # Get current version
    current_version = get_current_version()
    if current_version is None:
        lint_messages.append(
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="version-detection-failed",
                original=None,
                replacement=None,
                description="Could not determine current PyTorch version from version.txt",
            )
        )
        return lint_messages

    # Get versioned shim functions
    shim_functions = get_shim_functions()
    if not shim_functions:
        # No versioned functions found, nothing to check
        return lint_messages

    try:
        with open(filename) as f:
            lines = f.readlines()
    except Exception:
        return lint_messages

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
                                f"in a TORCH_FEATURE_VERSION block. This function requires:\n"
                                f"#if TORCH_FEATURE_VERSION >= {required_macro}\n"
                                f"  // ... your code calling {func_name} ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {required_macro}"
                            ),
                        )
                    )
                elif (
                    current_version_macro is not None
                    and current_version_macro != required_version
                ):
                    current_major, current_minor = current_version_macro
                    current_macro = f"TORCH_VERSION_{current_major}_{current_minor}_0"
                    lint_messages.append(
                        LintMessage(
                            path=filename,
                            line=line_num,
                            char=None,
                            code=LINTER_CODE,
                            severity=LintSeverity.ERROR,
                            name="wrong-version-for-shim-call",
                            original=None,
                            replacement=None,
                            description=(
                                f"Call to '{func_name}' is wrapped in {current_macro}, "
                                f"but this function requires {required_macro}. "
                                f"Please use the correct version guard:\n"
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
