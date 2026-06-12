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
    An identifier (name of function, typedef, class etc) that is used at a particular version.
    """

    identifier: str
    version: tuple[int, int, int] | None


class MultilineMatcher(NamedTuple):
    """
    Identifier matching using a start and end pattern, and a handler to extract the
    identifieruse from the accumulated buffer.
    """

    start_pattern: str | re.Pattern
    end_pattern: str | re.Pattern
    # Handler is a function that takes:
    # - buffer: str -> This is the accumulated buffer between the start and end
    #   pattern, unstripped and possibly containing newlines.
    # - version: tuple[int, int, int] | None -> Provides the version of the block this
    #   buffer is located in, None if outside of a versioning block.
    # It returns a list of IdentifierUse entries.
    handler: Callable[[str, tuple[int, int, int] | None], list[IdentifierUse]]


class IdentifierMatcher(NamedTuple):
    """
    Matcher that just searches for the pattern in each line, these patterns are not
    searched for when a multiline matcher is active.
    """

    pattern: str | re.Pattern
    identifier: str

    @staticmethod
    def word(word: str):
        return IdentifierMatcher(
            pattern=rf"\b{re.escape(word)}\b",
            identifier=word,
        )


def extract_factory(
    pattern,
) -> Callable[[str, tuple[int, int, int] | None], list[IdentifierUse]]:
    """
    Default handler that uses a single pattern and is expected to just find a single match.
    It uses re.DOTALL to compile the regex pattern provided.
    """
    p = re.compile(pattern, flags=re.DOTALL)

    def extractor(buffer: str, current_version: tuple[int, int, int] | None):
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
FUNCTION_IDENTIFIER_MATCHER = MultilineMatcher(
    start_pattern=r"\s*AOTI_TORCH_EXPORT",
    end_pattern=";",
    handler=extract_factory(r"AOTI_TORCH_EXPORT.+?(\w+)\s*\("),
)

# Also match typedef function pointers
TYPEDEF_IDENTIFIER_MATCHER = MultilineMatcher(
    start_pattern=r"\s*typedef",
    end_pattern=";",
    handler=extract_factory(r"typedef\s+.*\(\*(\w+)\)"),
)

# Match using declarations like: using TypeName = ...
USING_IDENTIFIER_MATCHER = MultilineMatcher(
    start_pattern=r"\s*using",
    end_pattern=";",
    handler=extract_factory(r"using\s+(\w+)\s*="),
)
# Match struct/class declarations like: struct StructName or class ClassName
STRUCT_CLASS_IDENTIFIER_MATCHER = MultilineMatcher(
    start_pattern=r"\s*(?:struct|class)",
    end_pattern=";",
    handler=extract_factory(r"(?:struct|class)\s+(\w+)"),
)


MULTILINE_MATCHERS = [
    FUNCTION_IDENTIFIER_MATCHER,
    TYPEDEF_IDENTIFIER_MATCHER,
    USING_IDENTIFIER_MATCHER,
    STRUCT_CLASS_IDENTIFIER_MATCHER,
]


def dynamic_call_parser(buffer: str, current_version: tuple[int, int, int] | None):
    pattern = r"TORCH_DYNAMIC_VERSION_CALL_(\d+)_(\d+)_(\d+)\(([^,]+),([^,\)]+)"
    buffer_without_space = buffer.replace(" ", "").replace("\n", "")
    res = re.findall(pattern, buffer_without_space)
    if not res:
        raise RuntimeError(
            f"Failed to parse dynamic version call pattern on buffer: {repr(buffer)}"
        )
    major, minor, patch = res[0][0:3]
    dynamic_version = (int(major), int(minor), int(patch))
    dynamic_lookup_identifier = res[0][3]
    fallback_identifier = res[0][4]
    return [
        IdentifierUse(dynamic_lookup_identifier, version=dynamic_version),
        IdentifierUse(fallback_identifier, version=current_version),
    ]


DYNAMIC_VERSION_CALL_IDENTIFIER_MATCHER = MultilineMatcher(
    start_pattern=r".*TORCH_DYNAMIC_VERSION_CALL_\d+_\d+_\d+",
    end_pattern=";",
    handler=dynamic_call_parser,
)


class MatcherAccumulator:
    """
    This class accumulates into a buffer whenever one of the start patterns of the matchers
    is encountered. When the end pattern is found it triggers the handler to extract the
    identifier(s) used in the buffer.

    It also serves as a matcher for identifiers, and searches each line outside of an active
    multiline matcher for identifiers.

    """

    def __init__(self, matchers: list[MultilineMatcher | IdentifierMatcher]):
        self._multi_line_matchers = []
        self._identifier_matchers = []

        # Compile all regexes and filter the matchers.
        for m in matchers:
            if isinstance(m, MultilineMatcher):
                end_pattern = re.compile(m.end_pattern)
                start_pattern = re.compile(m.start_pattern)
                self._multi_line_matchers.append(
                    MultilineMatcher(
                        start_pattern=start_pattern,
                        end_pattern=end_pattern,
                        handler=m.handler,
                    )
                )
            elif isinstance(m, IdentifierMatcher):
                pattern = re.compile(m.pattern)
                self._identifier_matchers.append(
                    IdentifierMatcher(pattern=pattern, identifier=m.identifier)
                )
        # Scope version is not part of reset, it persists through resets.
        self._scope_version = None
        self._reset_accumulation()

    def _reset_accumulation(self):
        """
        Resets the internal state such that new start patterns are sought.
        """
        self._buffer = ""
        self._end_token_found = False
        self._active_multiline_matcher = None
        self._found_identifiers = []

    def set_scope_version(self, scope_version: tuple[int, int, int] | None):
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
        if self._end_token_found:
            self._reset_accumulation()
        # New line, so clear the found identifiers.
        self._found_identifiers = []

        # If no matcher is active yet, check if any of them found a start token.
        if not self._active_multiline_matcher:
            for matcher in self._multi_line_matchers:
                found_start = matcher.start_pattern.finditer(line)
                for match in found_start:
                    self._active_multiline_matcher = matcher
                    line = line[match.start() :]
                    break
                if self._active_multiline_matcher:
                    break

        if self._active_multiline_matcher:
            # Ignore the part of the line that is commented because comments may have the end token in them.
            line = line[: line.find("//") if "//" in line else None]

            # See if the end token is present, if so strip the line down to just that segment.
            for match in self._active_multiline_matcher.end_pattern.finditer(line):
                line = line[: match.end()]
                self._end_token_found = True

            self._buffer += line

            # Now that the buffer is complete, parse it with the handler.
            if self._end_token_found:
                self._found_identifiers = self._active_multiline_matcher.handler(
                    self._buffer, self._scope_version
                )
        else:
            # No multiline matcher active, search using all the identifier matchers.
            for identifier_matcher in self._identifier_matchers:
                if identifier_matcher.pattern.search(line):
                    self._found_identifiers.append(
                        IdentifierUse(
                            identifier=identifier_matcher.identifier,
                            version=self._scope_version,
                        )
                    )

        return self._active_multiline_matcher is not None

    def identifiers_used(self) -> list[IdentifierUse]:
        """
        Returns identifiers found.
        """
        return self._found_identifiers


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
        self.preprocessor_stack: list[tuple[bool, tuple[int, int, int] | None]] = []

        # Current version requirement (if inside a version block)
        self.version_of_block: tuple[int, int, int] | None = None

        # Track if we're inside a block comment
        self.in_block_comment: bool = False

        # Regex to match version conditions in #if or #elif
        self.version_pattern = re.compile(
            r"#(?:if|elif)\s+TORCH_FEATURE_VERSION\s*>=\s*TORCH_VERSION_(\d+)_(\d+)_(\d+)"
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
                patch = int(version_match.group(3))
                version_tuple = (major, minor, patch)
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
                patch = int(version_match_elif.group(3))
                version_tuple = (major, minor, patch)
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
        found = self._identifier_accumulator.identifiers_used()
        return found if found is not None else []


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
