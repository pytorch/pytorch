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


# Add repo root to sys.path so we can import from tools
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.adapters.stable_shim_version_linter import PreprocessorTracker


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


def get_shim_functions(
    shim_files: list[Path | str] | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Extract function names from shim header files and their required version.
    Returns a dict mapping function name to (major, minor) version tuple.

    Only functions defined inside TORCH_FEATURE_VERSION blocks are extracted.
    Functions without version guards are ignored.

    Args:
        shim_files: List of paths to shim header files. If None, will use the default
                    paths to torch/csrc/stable/c/shim.h and
                    torch/csrc/inductor/aoti_torch/c/shim.h based on the repository root.
    """
    if shim_files is None:
        repo_root = Path(__file__).resolve().parents[3]
        shim_files_to_check = [
            repo_root / "torch/csrc/stable/c/shim.h",
            repo_root / "torch/csrc/inductor/aoti_torch/c/shim.h",
        ]
    else:
        shim_files_to_check = [Path(f) for f in shim_files]

    # Assert that all shim files exist
    missing_files = [f for f in shim_files_to_check if not f.exists()]
    if missing_files:
        raise RuntimeError(
            f"The following shim files do not exist: {missing_files}. "
            "Ensure all shim header files exist in the repository."
        )

    functions: dict[str, tuple[int, int]] = {}

    # Match function declarations like: AOTI_TORCH_EXPORT ... function_name(
    function_pattern = re.compile(r"AOTI_TORCH_EXPORT\s+\w+\s+(\w+)\s*\(")
    # Also match typedef function pointers
    typedef_pattern = re.compile(r"typedef\s+.*\(\*(\w+)\)")
    # Match using declarations like: using TypeName = ...
    using_pattern = re.compile(r"using\s+(\w+)\s*=")
    # Match struct/class declarations like: struct StructName or class ClassName
    struct_class_pattern = re.compile(r"(?:struct|class)\s+(\w+)")

    for shim_file in shim_files_to_check:
        with open(shim_file) as f:
            lines = f.readlines()

        tracker = PreprocessorTracker()

        for line in lines:
            is_directive_or_comment = tracker.process_line(line)

            # Only look for function declarations if not a comment/directive and inside a version block
            if not is_directive_or_comment:
                version_of_block = tracker.get_version_of_block()
                if version_of_block:
                    stripped = line.strip()
                    func_match = function_pattern.search(stripped)
                    if func_match:
                        func_name = func_match.group(1)
                        functions[func_name] = version_of_block
                        continue

                    typedef_match = typedef_pattern.search(stripped)
                    if typedef_match:
                        func_name = typedef_match.group(1)
                        functions[func_name] = version_of_block
                        continue

                    using_match = using_pattern.search(stripped)
                    if using_match:
                        type_name = using_match.group(1)
                        functions[type_name] = version_of_block
                        continue

                    struct_class_match = struct_class_pattern.search(stripped)
                    if struct_class_match:
                        type_name = struct_class_match.group(1)
                        functions[type_name] = version_of_block
                        continue

    if not functions:
        raise RuntimeError(
            "Could not find any versioned shim functions. "
            "Ensure at least one of the shim files exists and contains versioned functions."
        )

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
        f.write(
            "# If a function is not in this file, it was available before 2.10.0.\n"
        )
        f.write("# DO NOT EDIT MANUALLY.\n\n")

        for func_name, (major, minor) in sorted_functions:
            f.write(f"{func_name}: TORCH_VERSION_{major}_{minor}_0\n")


def check_file(
    filename: str, shim_functions: dict[str, tuple[int, int]]
) -> list[LintMessage]:
    """
    Check the input file for proper usage of versioned shim functions.

    Args:
        filename: File in torch/csrc/stable that calls functions from shim.
        shim_functions: Dictionary mapping function name to (major, minor) version tuple.
    """
    lint_messages: list[LintMessage] = []

    with open(filename) as f:
        lines = f.readlines()

    tracker = PreprocessorTracker()

    for line_num, line in enumerate(lines, 1):
        is_directive_or_comment = tracker.process_line(line)

        if is_directive_or_comment:
            continue

        version_of_block = tracker.get_version_of_block()

        for func_name, required_version in shim_functions.items():
            # Look for:
            # 1. Function calls like: func_name(
            # 2. Type usage like: func_name variable_name
            # Use word boundaries to avoid matching partial names

            if re.search(rf"\b{re.escape(func_name)}\b", line):
                major, minor = required_version
                required_macro = f"TORCH_VERSION_{major}_{minor}_0"

                if version_of_block is None:
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
                                f"Usage '{func_name}' from shim.h is not wrapped "
                                f"in a TORCH_FEATURE_VERSION block. This function requires at least:\n"
                                f"#if TORCH_FEATURE_VERSION >= {required_macro}\n"
                                f"  // ... your code calling {func_name} ...\n"
                                f"#endif // TORCH_FEATURE_VERSION >= {required_macro}"
                            ),
                        )
                    )
                elif version_of_block < required_version:
                    # Inside a version block, but version is too old
                    current_major, current_minor = version_of_block
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
                                f"Use of '{func_name}' is wrapped in {current_macro}, "
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
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    # Update the shim function versions file
    shim_functions = get_shim_functions()
    write_shim_function_versions(shim_functions)

    lint_messages = []
    for filename in args.filenames:
        lint_messages.extend(check_file(filename, shim_functions))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
