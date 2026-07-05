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
import sys
from pathlib import Path


# Add repo root to sys.path so we can import from tools
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.adapters._stable_shim_utils import (
    DYNAMIC_VERSION_CALL_IDENTIFIER_MATCHER,
    IdentifierMatcher,
    LintMessage,
    LintSeverity,
    MULTILINE_MATCHERS,
    PreprocessorTracker,
)


LINTER_CODE = "STABLE_SHIM_USAGE"


def get_shim_functions(
    shim_files: list[Path | str] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """
    Extract function names from shim header files and their required version.
    Returns a dict mapping function name to (major, minor) version tuple.

    Only functions defined inside TORCH_FEATURE_VERSION blocks are extracted.
    Functions without version guards are ignored.

    Args:
        shim_files: List of paths to shim header files. If None, will use the
                    default set of shim headers under torch/csrc/stable and
                    torch/csrc/inductor/aoti_torch (including generated shims)
    """
    if shim_files is None:
        repo_root = Path(__file__).resolve().parents[3]
        shim_files_to_check = [
            repo_root / "torch/csrc/stable/c/shim.h",
            repo_root / "torch/csrc/inductor/aoti_torch/c/shim.h",
            repo_root / "torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h",
            repo_root / "torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h",
            repo_root / "torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h",
            repo_root / "torch/csrc/inductor/aoti_torch/generated/c_shim_mps.h",
            repo_root / "torch/csrc/inductor/aoti_torch/generated/c_shim_xpu.h",
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

    identifiers: dict[str, tuple[int, int, int]] = {}

    for shim_file in shim_files_to_check:
        with open(shim_file) as f:
            lines = f.readlines()

        tracker = PreprocessorTracker(MULTILINE_MATCHERS)

        for line in lines:
            tracker.process_line(line)
            for identifier_version in tracker.identifiers_used():
                # Only look for function declarations if not a comment/directive and inside a version block
                if identifier_version.version is None:
                    continue
                identifiers[identifier_version.identifier] = identifier_version.version

    if not identifiers:
        raise RuntimeError(
            "Could not find any versioned shim functions. "
            "Ensure at least one of the shim files exists and contains versioned functions."
        )

    return identifiers


def write_shim_function_versions(
    functions: dict[str, tuple[int, int, int]],
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

        for func_name, (major, minor, patch) in sorted_functions:
            f.write(f"{func_name}: TORCH_VERSION_{major}_{minor}_{patch}\n")


def check_file(
    filename: str, shim_functions: dict[str, tuple[int, int, int]]
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

    # Generate the matchers from the provided function names.
    matchers = [DYNAMIC_VERSION_CALL_IDENTIFIER_MATCHER] + [
        IdentifierMatcher.word(function_name) for function_name in shim_functions
    ]

    tracker = PreprocessorTracker(matchers)

    for line_num, line in enumerate(lines, 1):
        if tracker.process_line(line):
            # Not regular code, no need to analyse it.
            continue

        for identifier_version in tracker.identifiers_used():
            version_of_block = identifier_version.version
            func_name = identifier_version.identifier
            required_version = shim_functions.get(func_name)
            if required_version is None:
                # Shim function has no version specified, so must always be available.
                continue

            major, minor, patch = required_version
            required_macro = f"TORCH_VERSION_{major}_{minor}_{patch}"

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
                current_major, current_minor, current_patch = version_of_block
                current_macro = (
                    f"TORCH_VERSION_{current_major}_{current_minor}_{current_patch}"
                )
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
