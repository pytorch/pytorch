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
from pathlib import Path


# Add repo root to sys.path so we can import from tools.setup_helpers
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.adapters._stable_shim_utils import (
    LintMessage,
    LintSeverity,
    PreprocessorTracker,
)
from tools.setup_helpers.gen_version_header import parse_version


LINTER_CODE = "STABLE_SHIM_VERSION"


def get_current_version() -> tuple[int, int]:
    """
    Get the current PyTorch version from version.txt.
    This uses the same logic as tools/setup_helpers/gen_version_header.py
    which is used to generate torch/headeronly/version.h from version.h.in.

    Returns (major, minor) tuple or None if not found.
    """
    repo_root = Path(__file__).resolve().parents[3]
    version_file = repo_root / "version.txt"

    if not version_file.exists():
        raise RuntimeError(
            "Could not find version.txt. This linter requires version.txt to run"
        )

    with open(version_file) as f:
        version = f.read().strip()
        major, minor, patch = parse_version(version)

    return (major, minor)


def get_added_lines(filename: str) -> set[int]:
    """
    Get the line numbers of added lines in:
    1. Current uncommitted changes (git diff HEAD)
    2. All commits in the current PR (git diff merge-base..HEAD)

    This ensures that in CI we catch version macro issues across all PR commits.

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

        # Get merge-base with origin/main to check all PR commits
        result = subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to fetch origin. Error: {result.stderr.strip()}"
            )

        result = subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to find merge-base with origin/main. "
                f"Make sure origin/main exists (run 'git fetch origin main'). "
                f"Error: {result.stderr.strip()}"
            )

        merge_base = result.stdout.strip()
        result = subprocess.run(
            ["git", "diff", f"{merge_base}..HEAD", filename],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get git diff information for {filename}. Error: {result.stderr}"
            )
        added_lines.update(parse_diff(result.stdout))

    except Exception as e:
        raise RuntimeError(
            f"Failed to get git diff information for {filename}. Error: {e}"
        ) from e

    return added_lines


def check_file(filename: str) -> list[LintMessage]:
    """
    Parse the stable/c/shim.h file and check that:
    1. All function declarations are within TORCH_FEATURE_VERSION blocks
    2. New functions added in this commit use the current version macro

    For the AOTI shim (torch/csrc/inductor/aoti_torch/c/shim.h), we only
    enforce versioning on NEW function declarations, since existing functions
    are intentionally not version-guarded.
    """
    lint_messages: list[LintMessage] = []

    # Check if this is the AOTI shim - only enforce versioning on new lines
    is_aoti_shim = "torch/csrc/inductor/aoti_torch/c/shim.h" in filename

    # Get current version
    current_version = get_current_version()
    major, minor = current_version
    expected_version_macro = f"TORCH_VERSION_{major}_{minor}_0"
    expected_version_check = f"#if TORCH_FEATURE_VERSION >= {expected_version_macro}"

    # Get lines that are uncommitted or added in the most recent commit
    added_lines = get_added_lines(filename)

    with open(filename) as f:
        lines = f.readlines()

    # Use PreprocessorTracker to handle preprocessor directives
    tracker = PreprocessorTracker()

    # Track extern "C" blocks separately
    inside_extern_c = False

    # Patterns for extern "C" blocks
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

        # Skip empty lines
        if not stripped:
            continue

        # Let the tracker process preprocessor directives and comments
        is_directive_or_comment = tracker.process_line(line)

        if is_directive_or_comment:
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

                # Get current version state from tracker
                inside_version_block = tracker.is_in_version_block()
                tracker_version = tracker.get_version_of_block()
                version_of_block_macro = (
                    f"TORCH_VERSION_{tracker_version[0]}_{tracker_version[1]}_0"
                    if tracker_version
                    else None
                )

                if not inside_version_block:
                    # Function declaration outside of version block
                    if not is_new_line:
                        # Existing function declaration outside of version block in aoti shim is ignored
                        if is_aoti_shim:
                            continue
                        expected_version_macro_str = "TORCH_VERSION_X_Y_Z"
                        expected_version_check_str = (
                            f"#if TORCH_FEATURE_VERSION >= {expected_version_macro}"
                        )
                        additional_text = "\nX, Y, and Z correspond to the TORCH_ABI_VERSION when the function was added."
                    else:
                        expected_version_macro_str = expected_version_macro
                        expected_version_check_str = expected_version_check
                        additional_text = ""
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
                                f"{expected_version_check_str}\n"
                                f"// ... your declarations ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {expected_version_macro_str}"
                                f"{additional_text}"
                            ),
                        )
                    )
                elif is_new_line and version_of_block_macro != expected_version_macro:
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
                                f"but is wrapped in {version_of_block_macro}. "
                                f"New additions in this commit must use the current version:\n"
                                f"{expected_version_check}\n"
                                f"// ... your declarations ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {expected_version_macro}"
                            ),
                        )
                    )

    return lint_messages


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
