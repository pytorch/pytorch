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
from typing import NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


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


class IdentifierUse(NamedTuple):
    """
    An identifier (name of function, typedef, class etc) that at a particular version.
    """

    identifier: str
    version: tuple[int, int] | None


class IdentifierMatcher(NamedTuple):
    """
    Identifier matching using a start and optional end pattern, and a handler to extract the
    identifieruse from the accumulated buffer.
    If the end pattern is empty, the handler is triggered immediately after the start pattern.
    """

    start_pattern: str | re.Pattern
    end_pattern: str | re.Pattern | None
    # Handler is a function that takes:
    # - buffer: str -> This is the accumulated buffer between the start and end
    #   pattern, unstripped and possibly containing newlines.
    # - version: tuple[int, int] | None -> Provides the version of the block this
    #   buffer is located in, None if outside of a versioning block.
    # It returns a list of IdentifierUse entries.
    handler: Callable[[str, tuple[int, int] | None], list[IdentifierUse]]


def extract_factory(
    pattern,
) -> Callable[[str, tuple[int, int] | None], list[IdentifierUse]]:
    """
    Default handler that uses a single pattern and is expected to just find a single match.
    It uses re.DOTALL to compile the regex pattern provided.
    """
    p = re.compile(pattern, flags=re.DOTALL)

    def extractor(buffer: str, current_version: tuple[int, int] | None):
        identifier = p.search(buffer)

        if identifier:
            return [
                IdentifierUse(identifier=identifier.group(1), version=current_version)
            ]

        # If no identifier was found, this is a bug in the matcher as it means the extractor
        # did not find anything between the start and end pattern.
        raise ValueError(
            "extractor didn't find results, mismatch between start-end patterns and handler"
        )

    return extractor


# When adding a matcher, add a test to tools/test/test_stable_shim_utils.py
# to verify it works as expected, this test file is also helpful for iterating
# on the regular expressions.

# Match function declarations like: AOTI_TORCH_EXPORT ... function_name(
FUNCTION_IDENTIFIER_MATCHER = IdentifierMatcher(
    start_pattern=r"\s*AOTI_TORCH_EXPORT",
    end_pattern=";",
    handler=extract_factory(r"AOTI_TORCH_EXPORT.+?(\w+)\s*\("),
)

# Also match typedef function pointers
TYPEDEF_IDENTIFIER_MATCHER = IdentifierMatcher(
    start_pattern=r"\s*typedef",
    end_pattern=";",
    handler=extract_factory(r"typedef\s+.*\(\*(\w+)\)"),
)

# Match using declarations like: using TypeName = ...
USING_IDENTIFIER_MATCHER = IdentifierMatcher(
    start_pattern=r"\s*using",
    end_pattern=";",
    handler=extract_factory(r"using\s+(\w+)\s*="),
)
# Match struct/class declarations like: struct StructName or class ClassName
STRUCT_CLASS_IDENTIFIER_MATCHER = IdentifierMatcher(
    start_pattern=r"\s*(?:struct|class)",
    end_pattern=";",
    handler=extract_factory(r"(?:struct|class)\s+(\w+)"),
)


def arbitrary_identifier_matcher(pattern):
    """
    Create an arbitrary identifier matcher, this is what is used to find the
    relevant identifiers in normal code.
    """

    def extractor(
        buffer: str, current_version: tuple[int, int] | None
    ) -> list[IdentifierUse]:
        return [IdentifierUse(identifier=pattern, version=current_version)]

    # Use word boundaries to avoid matching partial names
    # if re.search(rf"\b{re.escape(func_name)}\b", line):
    return IdentifierMatcher(
        start_pattern=rf"\b{re.escape(pattern)}\b",
        end_pattern=None,
        handler=extractor,
    )


IDENTIFIER_MATCHERS = [
    FUNCTION_IDENTIFIER_MATCHER,
    TYPEDEF_IDENTIFIER_MATCHER,
    USING_IDENTIFIER_MATCHER,
    STRUCT_CLASS_IDENTIFIER_MATCHER,
]


class MatcherAccumulator:
    """
    This class accumulates into a buffer whenever one of the start patterns of the matchers
    is encountered. When the end pattern is found it triggers the handler to extract the
    identifier(s) used in the buffer.

    After the handler is triggered, the internal state is reset with reset() and it continues
    scanning lines for new start patterns.
    """

    def __init__(self, matchers: list[IdentifierMatcher]):
        self._matchers = []

        # Compile all regexes.
        for m in matchers:
            end_pattern = re.compile(m.end_pattern) if m.end_pattern else None
            start_pattern = re.compile(m.start_pattern)
            self._matchers.append(
                IdentifierMatcher(
                    start_pattern=start_pattern,
                    end_pattern=end_pattern,
                    handler=m.handler,
                )
            )
        # Scope version is not part of reset, it persists through resets.
        self._scope_version = None
        self._reset()

    def _reset(self):
        """
        Resets the internal state such that new start patterns are sought.
        """
        self._buffer = ""
        self._end_token_found = False
        self._active_matcher = None
        self._found = []

    def set_scope_version(self, scope_version: tuple[int, int] | None):
        """
        Function called whenever the outer version 'scope' we're parsing from
        changes, this is provided to the handler.
        """
        self._scope_version = scope_version

    def process_line(
        self,
        line: str,
    ) -> bool:
        """
        Processes a single line, searches for start and end token and strips out single-line comments.

        Returns whether this line is part of an actively being parsed matcher.
        """
        if self._found:
            self._reset()

        # If no matcher is active yet, check if any of them found a start token.
        if not self._active_matcher:
            for matcher in self._matchers:
                found_start = matcher.start_pattern.finditer(line)
                for match in found_start:
                    if matcher.end_pattern is None:
                        # No end pattern, single match handling, immediately invoke handler on this section.
                        self._found += matcher.handler(line, self._scope_version)
                        self._end_token_found = True
                    else:
                        # Looking for end pattern, stirp line for addition to buffer and store matcher.
                        line = line[match.start() :]
                        self._active_matcher = matcher
                    break
                if self._active_matcher:
                    # Only one multi line matcher can be active at one time, stop searching for matches.
                    break

        if self._active_matcher:
            # First see if the end token is present, if so strip the line down to just that segment.
            for match in self._active_matcher.end_pattern.finditer(line):
                line = line[: match.end()]
                self._end_token_found = True

            # Ignore the part of the line that is commented because comments may have the end token in them.
            self._buffer += line[: line.find("//") if "//" in line else None]

            # If an end token was found, add it to the found matches.
            if self._end_token_found:
                self._found += self._active_matcher.handler(
                    self._buffer, self._scope_version
                )

        return self._active_matcher is not None

    def identifiers_used(self) -> list[IdentifierUse]:
        """
        Returns the identifiers used in the pattern as parsed, identifiers are retrieved
        at the end of the pattern, they are cleared when the next line is processed.
        """
        return self._found


class PreprocessorTracker:
    """
    Helper class to track preprocessor directives and version blocks.

    This class maintains state as it processes C/C++ preprocessor directives
    (#if, #elif, #else, #endif) and tracks which code is inside version blocks.

    While processing the lines, it does keep an internal buffer to accumulate
    information from multiple lines, this is mostly important when identifying
    the version required for identifiers. An example are exported functions,
    they start with `AOTI_TORCH_EXPORT` and end with `;`,  but those two
    strings may be on separate lines. The function name itself cannot be split
    over multiple lines.
    """

    def __init__(self, matchers):
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

        self._identifier_accumulator = MatcherAccumulator(matchers)

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

        # This is a line with code, so process it with the matcher.
        self._identifier_accumulator.set_scope_version(self.version_of_block)
        self._identifier_accumulator.process_line(line)

        # Not a preprocessor directive or comment
        return False

    def identifiers_used(self) -> list[IdentifierUse]:
        return self._identifier_accumulator.identifiers_used()


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
