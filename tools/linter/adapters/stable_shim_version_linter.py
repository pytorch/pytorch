#!/usr/bin/env python3
"""
STABLE_SHIM_VERSION: Ensures that function declarations in stable/c/shim.h
are properly wrapped in TORCH_FEATURE_VERSION macros corresponding to the
current TORCH_ABI_VERSION.
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


LINTER_CODE = "STABLE_SHIM_VERSION"


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
    Get the line numbers of added lines in:
    1. Current uncommitted changes (git diff HEAD)
    2. The most recent commit (git diff HEAD~1..HEAD)

    This ensures that even if someone commits locally before running the linter,
    we still catch version macro issues in their recent changes.

    Returns:
        Set of line numbers (1-indexed) that are new additions.
    """
    import subprocess

    added_lines = set()

    def parse_diff(diff_output: str) -> set[int]:
        """Parse git diff output and return line numbers of added lines."""
        lines = set()
        current_line = 0
        for line in diff_output.split("\n"):
            # Unified diff format: @@ -old_start,old_count +new_start,new_count @@
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                # This is an added line
                lines.add(current_line)
                current_line += 1
            elif not line.startswith("-"):
                # Context line or unchanged line
                current_line += 1
        return lines

    try:
        # Check uncommitted changes (working directory vs HEAD)
        result = subprocess.run(
            ["git", "diff", "HEAD", filename],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            added_lines.update(parse_diff(result.stdout))

        # Also check the most recent commit (HEAD vs HEAD~1)
        # This catches cases where someone commits before running the linter
        result = subprocess.run(
            ["git", "diff", "HEAD~1..HEAD", filename],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            added_lines.update(parse_diff(result.stdout))

    except Exception as e:
        raise RuntimeError(
            f"Failed to get git diff information for {filename}. Error: {e}"
        ) from e

    return added_lines


def parse_shim_file(filename: str) -> list[LintMessage]:
    """
    Parse the stable/c/shim.h file and check that:
    1. All function declarations are within TORCH_FEATURE_VERSION blocks
    2. New functions added in this commit use the current version macro
    """
    lint_messages: list[LintMessage] = []

    # Get current version
    current_version = get_current_version()
    if current_version is None:
        raise RuntimeError(
            "Could not determine current PyTorch version from version.txt. "
            "This linter requires version.txt to exist in the repository root and contain a valid version number."
        )

    major, minor = current_version
    expected_version_macro = f"TORCH_VERSION_{major}_{minor}_0"
    expected_version_check = f"#if TORCH_FEATURE_VERSION >= {expected_version_macro}"

    # Get lines that are uncommitted or added in the most recent commit
    added_lines = get_added_lines(filename)

    with open(filename) as f:
        lines = f.readlines()

    # Track state
    inside_version_block = False
    current_version_macro = None
    inside_extern_c = False

    # Track ALL preprocessor conditional blocks to properly match #if/#endif pairs
    # Each element is (is_version_block, version_macro_or_none)
    preprocessor_stack: list[tuple[bool, str | None]] = []

    # Patterns
    version_start_pattern = re.compile(
        r"#if\s+TORCH_FEATURE_VERSION\s*>=\s*(TORCH_VERSION_\d+_\d+_\d+)"
    )
    extern_c_pattern = re.compile(r'extern\s+"C"\s*{')
    extern_c_end_pattern = re.compile(r'}\s*//\s*extern\s+"C"')

    # Function declaration patterns - looking for AOTI_TORCH_EXPORT or typedef
    function_decl_patterns = [
        re.compile(r"^\s*AOTI_TORCH_EXPORT\s+\w+"),  # AOTI_TORCH_EXPORT functions
        re.compile(r"^\s*typedef\s+.*\(\*\w+\)"),  # typedef function pointers
        re.compile(r"^\s*using\s+\w+\s*="),  # using declarations
    ]

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("//"):
            continue

        # Check for TORCH_FEATURE_VERSION block start
        version_match = version_start_pattern.match(stripped)
        if version_match:
            version_macro = version_match.group(1)
            preprocessor_stack.append((True, version_macro))
            inside_version_block = True
            current_version_macro = version_macro
            continue

        # Track any other #if/#ifdef/#ifndef directives
        if stripped.startswith(("#if", "#ifdef", "#ifndef")) and not version_match:
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

        # Skip other preprocessor directives
        if stripped.startswith("#"):
            continue

        # Track extern "C" blocks
        if extern_c_pattern.search(stripped):
            inside_extern_c = True
            continue
        if extern_c_end_pattern.search(stripped):
            inside_extern_c = False
            continue

        # Check for function declarations
        if inside_extern_c:
            is_function_decl = any(
                pattern.match(stripped) for pattern in function_decl_patterns
            )

            if is_function_decl:
                # Check if this is a newly added line
                is_new_line = line_num in added_lines

                if not inside_version_block:
                    # Function declaration outside of version block
                    lint_messages.append(
                        LintMessage(
                            path=filename,
                            line=line_num,
                            char=None,
                            code=LINTER_CODE,
                            severity=LintSeverity.ERROR,
                            name="unversioned-function-declaration",
                            original=None,
                            replacement=None,
                            description=(
                                f"Function declaration found outside of TORCH_FEATURE_VERSION block. "
                                f"All function declarations must be wrapped in:\n"
                                f"{expected_version_check}\n"
                                f"// ... your declarations ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {expected_version_macro}"
                            ),
                        )
                    )
                elif is_new_line and current_version_macro != expected_version_macro:
                    # New function declaration using wrong version macro
                    lint_messages.append(
                        LintMessage(
                            path=filename,
                            line=line_num,
                            char=None,
                            code=LINTER_CODE,
                            severity=LintSeverity.ERROR,
                            name="wrong-version-for-new-function",
                            original=None,
                            replacement=None,
                            description=(
                                f"New function declaration should use {expected_version_macro}, "
                                f"but is wrapped in {current_version_macro}. "
                                f"New additions in this commit must use the current version:\n"
                                f"{expected_version_check}\n"
                                f"// ... your declarations ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {expected_version_macro}"
                            ),
                        )
                    )

    return lint_messages


def check_file(filename: str) -> list[LintMessage]:
    """
    Check if the file is stable/c/shim.h and lint it.
    """
    # Only lint the specific file
    if not filename.endswith("torch/csrc/stable/c/shim.h"):
        return []

    return parse_shim_file(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="stable shim version linter",
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
