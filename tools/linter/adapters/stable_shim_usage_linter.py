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


class PreprocessorTracker:
    """
    Helper class to track preprocessor directives and version blocks.

    This class maintains state as it processes C/C++ preprocessor directives
    (#if, #elif, #else, #endif) and tracks which code is inside version blocks.
    """

    def __init__(self):
        """Initialize the preprocessor tracker."""
        # Stack of (is_version_block, version_tuple) tuples
        # is_version_block: True if this is a TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_0 block
        # version_tuple: (major, minor) if is_version_block is True, else None
        self.preprocessor_stack: list[tuple[bool, tuple[int, int] | None]] = []

        # Current version requirement (if inside a version block)
        self.current_version: tuple[int, int] | None = None

        # Track if we're inside a block comment
        self.in_block_comment: bool = False

        # Regex to match version conditions in #if or #elif
        self.version_pattern = re.compile(
            r"#(?:if|elif)\s+TORCH_FEATURE_VERSION\s*>=\s*TORCH_VERSION_(\d+)_(\d+)_\d+"
        )

    def process_line(self, line: str) -> bool:
        """
        Process a line and update the preprocessor state.

        Args:
            line: The line to process

        Returns:
            True if the line was processed (is a preprocessor directive or comment),
            False if it's a regular code line that should be further analyzed.
        """
        stripped = line.strip()

        # Handle block comments (/* ... */)
        # Check if we're entering a block comment
        if "/*" in line:
            self.in_block_comment = True

        # If we're in a block comment, check if we're exiting
        if self.in_block_comment:
            if "*/" in line:
                self.in_block_comment = False
            return True  # Skip the line if we're in a block comment

        # Skip line comments - they're not active code
        if stripped.startswith("//"):
            return True

        # Track #if directives
        if stripped.startswith("#if"):
            version_match = self.version_pattern.match(stripped)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                version_tuple = (major, minor)
                self.preprocessor_stack.append((True, version_tuple))
                self.current_version = version_tuple
            else:
                # Regular #if (not a version block)
                self.preprocessor_stack.append((False, None))
            return True

        # Track #ifdef and #ifndef directives (not version blocks)
        if stripped.startswith(("#ifdef", "#ifndef")):
            self.preprocessor_stack.append((False, None))
            return True

        # Track #endif directives
        if stripped.startswith("#endif"):
            if self.preprocessor_stack:
                is_version_block, _ = self.preprocessor_stack.pop()
                if is_version_block:
                    # Restore previous version block if any
                    self.current_version = None
                    for i in range(len(self.preprocessor_stack) - 1, -1, -1):
                        if self.preprocessor_stack[i][0]:
                            self.current_version = self.preprocessor_stack[i][1]
                            break
            return True

        # Track #else directives
        # #else replaces the previous #if or #elif, so we pop and push
        if stripped.startswith("#else"):
            if self.preprocessor_stack:
                self.preprocessor_stack.pop()
            # #else is never versioned, so push (False, None)
            self.preprocessor_stack.append((False, None))
            self.current_version = None
            return True

        # Track #elif directives
        # #elif replaces the previous #if or #elif, so we pop and push
        if stripped.startswith("#elif"):
            if self.preprocessor_stack:
                self.preprocessor_stack.pop()

            self.current_version = None

            # Check if this #elif has a version condition
            version_match_elif = self.version_pattern.match(stripped)
            if version_match_elif:
                major = int(version_match_elif.group(1))
                minor = int(version_match_elif.group(2))
                version_tuple = (major, minor)
                self.preprocessor_stack.append((True, version_tuple))
                self.current_version = version_tuple
            else:
                # Not a version elif, treat as regular conditional
                self.preprocessor_stack.append((False, None))
            return True

        # Not a preprocessor directive or comment
        return False

    def is_in_version_block(self) -> bool:
        """Check if currently inside any version block."""
        return self.current_version is not None

    def get_current_version(self) -> tuple[int, int] | None:
        """Get the current version requirement, or None if not in a version block."""
        return self.current_version


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


def get_shim_functions(
    shim_file: Path | str | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Extract function names from shim.h and their required version.
    Returns a dict mapping function name to (major, minor) version tuple.

    Functions defined inside TORCH_FEATURE_VERSION blocks require that version.

    Args:
        shim_file: Path to the shim.h file. If None, will compute the default path
                   to torch/csrc/stable/c/shim.h based on the repository root.
    """
    if shim_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        shim_file = repo_root / "torch/csrc/stable/c/shim.h"
    else:
        shim_file = Path(shim_file)

    if not shim_file.exists():
        raise RuntimeError(
            f"Could not find shim.h at {shim_file}. "
            f"This linter requires torch/csrc/stable/c/shim.h to exist in the repository."
        )

    functions: dict[str, tuple[int, int]] = {}

    with open(shim_file) as f:
        lines = f.readlines()

    # Use PreprocessorTracker to track version blocks
    tracker = PreprocessorTracker()

    # Match function declarations like: AOTI_TORCH_EXPORT ... function_name(
    function_pattern = re.compile(r"AOTI_TORCH_EXPORT\s+\w+\s+(\w+)\s*\(")
    # Also match typedef function pointers
    typedef_pattern = re.compile(r"typedef\s+.*\(\*(\w+)\)")

    for line in lines:
        # Process line with tracker - returns True if it's a comment or preprocessor directive
        is_directive_or_comment = tracker.process_line(line)

        # Only look for function declarations if not a comment/directive and inside a version block
        if not is_directive_or_comment:
            current_version = tracker.get_current_version()
            if current_version:
                stripped = line.strip()
                func_match = function_pattern.search(stripped)
                if func_match:
                    func_name = func_match.group(1)
                    functions[func_name] = current_version
                    continue

                typedef_match = typedef_pattern.search(stripped)
                if typedef_match:
                    func_name = typedef_match.group(1)
                    functions[func_name] = current_version
                    continue

    return functions


def write_shim_function_versions(
    functions: dict[str, tuple[int, int]],
    output_file: Path | str | None = None,
) -> None:
    """
    Write the shim function versions to a text file.

    Args:
        functions: Dictionary mapping function name to (major, minor) version tuple.
        output_file: Path to the output file. If None, will write to
                     torch/csrc/stable/c/shim_function_versions.txt in the repository.
    """
    if output_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        output_file = repo_root / "torch/csrc/stable/c/shim_function_versions.txt"
    else:
        output_file = Path(output_file)

    # Sort functions by version, then by name for consistency
    sorted_functions = sorted(functions.items(), key=lambda x: (x[1], x[0]))

    with open(output_file, "w") as f:
        f.write(
            "# Auto-generated file listing shim functions and their minimum required versions\n"
        )
        f.write("# Format: function_name: TORCH_VERSION_MAJOR_MINOR_PATCH\n")
        f.write("#\n")
        f.write(
            "# This file is automatically updated by the stable_shim_usage_linter.\n"
        )
        f.write("# DO NOT EDIT MANUALLY.\n\n")

        for func_name, (major, minor) in sorted_functions:
            f.write(f"{func_name}: TORCH_VERSION_{major}_{minor}_0\n")


def check_file(filename: str) -> list[LintMessage]:
    """
    Check if the file is in torch/csrc/stable and lint it for proper
    usage of versioned shim functions.
    """
    lint_messages: list[LintMessage] = []

    # Get versioned shim functions
    shim_functions = get_shim_functions()
    if not shim_functions:
        raise RuntimeError("Could not extract any shim_functions")

    with open(filename) as f:
        lines = f.readlines()

    # Use PreprocessorTracker to track version blocks
    tracker = PreprocessorTracker()

    for line_num, line in enumerate(lines, 1):
        # Process line with tracker - returns True if it's a comment or preprocessor directive
        is_directive_or_comment = tracker.process_line(line)

        # Skip if it's a comment or directive - only check regular code lines
        if is_directive_or_comment:
            continue

        # Check for calls to versioned shim functions
        current_version = tracker.get_current_version()

        for func_name, required_version in shim_functions.items():
            # Look for:
            # 1. Function calls like: func_name(
            # 2. Type usage like: func_name variable_name
            # Use word boundaries to avoid matching partial names

            # Check for function calls or type usage
            if re.search(rf"\b{re.escape(func_name)}\b", line):
                major, minor = required_version
                required_macro = f"TORCH_VERSION_{major}_{minor}_0"

                if current_version is None:
                    # Not inside any version block
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
                elif current_version < required_version:
                    # Inside a version block, but version is too old
                    current_major, current_minor = current_version
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

    # Update the shim function versions file
    shim_functions = get_shim_functions()
    write_shim_function_versions(shim_functions)

    lint_messages = []
    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
