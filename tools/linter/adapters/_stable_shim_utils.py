"""
Shared utilities for stable-shim linters.

Consumed by:
    - tools/linter/adapters/stable_shim_version_linter.py
    - tools/linter/adapters/stable_shim_usage_linter.py
    - tools/linter/adapters/generated_shims_version_linter.py
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from pathlib import Path
from typing import cast, NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.setup_helpers.gen_version_header import parse_version


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


class DynamicVersionCall(NamedTuple):
    dynamic_call_version: tuple[int, int]
    fallback_function: str
    target_function: str


class DynamicVersionCallParser:
    def __init__(self):
        self.dynamic_version_call_pattern = re.compile(
            r"TORCH_DYNAMIC_VERSION_CALL_(\d+)_(\d+)_\d+\("
        )
        self.reset()

    def reset(self):
        # Version of the dynamic version call itself, which must match the target function.
        self.call_version: tuple[int, int] | None = None
        # Holds the function names of the first two arguments of the macro, first is the fallback, second the target.
        self.function_names: list[str] = []
        # Whether all necessary data is available.
        self.dynamic_version_call_ready: bool = False
        # Whether we are currently accumulating data while in a call.
        self.in_dynamic_version_call: bool = False
        # Buffer to accumulate lines in to parse function names from.
        self.argument_contents = ""

    def process_line(self, line: str) -> bool:
        if self.dynamic_version_call_ready:
            self.reset()

        start_dynamic_version_call = self.dynamic_version_call_pattern.finditer(line)
        for match in start_dynamic_version_call:
            self.in_dynamic_version_call = True
            self.call_version = (int(match.group(1)), int(match.group(2)))
            # Strip the part of the line that started the macro call.
            line = line[match.end() :]

        if self.in_dynamic_version_call:
            # Ignore the part of the line that is commented because comments may have ')' and ',' in them.
            self.argument_contents += line[: line.find("//")]

        # Try to obtain information from the argument contents we've collected so far.
        # We only need to match two situations;
        #  TORCH_DYNAMIC_VERSION_CALL_2_13_0(A, B, ...)
        #  TORCH_DYNAMIC_VERSION_CALL_2_13_0(A, B)
        # Try to find both arguments;
        arguments = []
        for match in re.finditer("([^,]+)[,\\)]", self.argument_contents):
            arguments.append(match.group(1).strip())
            if len(arguments) == 2:
                break

        # If we have two arguments now, mark the parsing as complete.
        if len(arguments) == 2:
            self.function_names = arguments
            self.dynamic_version_call_ready = True

        return self.in_dynamic_version_call

    def information(self) -> None | DynamicVersionCall:
        if not self.dynamic_version_call_ready:
            return None

        # Cast is safe because version is parsed as pattern detect and must
        # be set when the call if ready.
        dynamic_call_version = cast(tuple[int, int], self.call_version)
        return DynamicVersionCall(
            dynamic_call_version=dynamic_call_version,
            fallback_function=self.function_names[0],
            target_function=self.function_names[1],
        )

    def is_ready(self) -> bool:
        return self.dynamic_version_call_ready


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
        self.version_of_block: tuple[int, int] | None = None

        # Track if we're inside a block comment
        self.in_block_comment: bool = False

        # Regex to match version conditions in #if or #elif
        self.version_pattern = re.compile(
            r"#(?:if|elif)\s+TORCH_FEATURE_VERSION\s*>=\s*TORCH_VERSION_(\d+)_(\d+)_\d+"
        )

        # Parser for the dynamic version call, since this spans multiple lines we need to accumulate some state.
        self.dynamic_version_call_parser = DynamicVersionCallParser()

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
                self.version_of_block = version_tuple
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
                    self.version_of_block = None
                    for i in range(len(self.preprocessor_stack) - 1, -1, -1):
                        if self.preprocessor_stack[i][0]:
                            self.version_of_block = self.preprocessor_stack[i][1]
                            break
            return True

        # Track #else directives
        # #else replaces the previous #if or #elif, so we pop and push
        if stripped.startswith("#else"):
            if self.preprocessor_stack:
                self.preprocessor_stack.pop()
            # #else is never versioned, so push (False, None)
            self.preprocessor_stack.append((False, None))
            self.version_of_block = None
            return True

        # Track #elif directives
        # #elif replaces the previous #if or #elif, so we pop and push
        if stripped.startswith("#elif"):
            if self.preprocessor_stack:
                self.preprocessor_stack.pop()

            self.version_of_block = None

            # Check if this #elif has a version condition
            version_match_elif = self.version_pattern.match(stripped)
            if version_match_elif:
                major = int(version_match_elif.group(1))
                minor = int(version_match_elif.group(2))
                version_tuple = (major, minor)
                self.preprocessor_stack.append((True, version_tuple))
                self.version_of_block = version_tuple
            else:
                # Not a version elif, treat as regular conditional
                self.preprocessor_stack.append((False, None))
            return True

        in_dynamic_section = self.dynamic_version_call_parser.process_line(line)
        if in_dynamic_section:
            # Skip lines while the dynamic version call parser is consuming lines.
            return True

        # Not a preprocessor directive or comment
        return False

    def is_in_version_block(self) -> bool:
        """Check if currently inside any version block."""
        return self.version_of_block is not None

    def get_version_of_block(self) -> tuple[int, int] | None:
        """Get the current version requirement, or None if not in a version block."""
        return self.version_of_block

    def get_dynamic_call_information(self) -> DynamicVersionCall | None:
        if not self.dynamic_version_call_parser.is_ready():
            return None

        return self.dynamic_version_call_parser.information()


def get_current_version() -> tuple[int, int, int]:
    """
    Get the current PyTorch version from version.txt. Returns (major, minor, patch) tuple.
    """
    version_file = REPO_ROOT / "version.txt"

    if not version_file.exists():
        raise RuntimeError(
            "Could not find version.txt. This linter requires version.txt to run"
        )

    with open(version_file) as f:
        version = f.read().strip()
        major, minor, patch = parse_version(version)

    return (major, minor, patch)


if __name__ == "__main__":
    with open(
        Path(__file__).parent.parent.parent
        / "test/stable_shim_usage_linter_data/sample_usage.h"
    ) as f:
        lines = f.readlines()

    parser = PreprocessorTracker()
    for i, line in enumerate(lines):
        to_handle = parser.process_line(line)
        print(f"to handle: {to_handle} with {i} line: {line}")
        dynamic_call_info = parser.get_dynamic_call_information()
        if dynamic_call_info:
            print(dynamic_call_info)
