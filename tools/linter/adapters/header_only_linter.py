#!/usr/bin/env python3
"""
Checks that all symbols in torch/header_only_apis.txt are tested in a .cpp
test file to ensure header-only-ness. The .cpp test file must be built
without linking libtorch.

Also checks the converse: that every listed symbol is still reachable from the
header it is filed under, so that deleting an API does not silently leave a
stale entry advertising it.
"""

import argparse
import functools
import json
import re
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "HEADER_ONLY_LINTER"


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


CPP_TEST_GLOBS = [
    "test/cpp/aoti_abi_check/*.cpp",
    "test/cpp/aoti_abi_check/cuda/*.cu",
]

REPO_ROOT = Path(__file__).parents[3]

# A comment naming the header(s) the symbols below it come from, e.g.
#   "# torch/headeronly/util/Half.h"
#   "# c10/util/complex.h, torch/headeronly/util/complex.h"
# Other comments in the file are prose and do not change the attribution.
HEADER_COMMENT = re.compile(r"^#\s*([\w/.]+\.h(?:\s*,\s*[\w/.]+\.h)*)\s*$")
IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


class Symbol(NamedTuple):
    name: str
    lineno: int
    headers: list[str]


@functools.cache
def _read(path: Path) -> str:
    return path.read_text(errors="replace")


@functools.cache
def resolve_header(include: str) -> Path | None:
    """Map an include path to a file on disk. ATen lives under aten/src."""
    for candidate in (REPO_ROOT / include, REPO_ROOT / "aten/src" / include):
        if candidate.is_file():
            return candidate
    return None


def include_closure(header: Path) -> set[Path]:
    """Every in-tree header reachable from `header`.

    What matters is whether including the filed header gets you the symbol,
    not whether the token sits in that exact file. ATen/cpu/vec/vec.h is a
    small umbrella over the arch-specific vec headers, and Dispatch_v2.h pulls
    its macros in from Dispatch.h.
    """
    seen = {header}
    queue = [header]
    while queue:
        for include in INCLUDE.findall(_read(queue.pop())):
            nxt = resolve_header(include)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def find_matched_symbols(
    symbols_regex: re.Pattern[str], test_globs: list[str] = CPP_TEST_GLOBS
) -> set[str]:
    """
    Goes through all lines not starting with // in the cpp files and
    accumulates a list of matches with the symbols_regex. Note that
    we expect symbols_regex to be sorted in reverse alphabetical
    order to allow superset regexes to get matched.
    """
    matched_symbols = set()
    # check noncommented out lines of the test files
    for cpp_test_glob in test_globs:
        for test_file in REPO_ROOT.glob(cpp_test_glob):
            with open(test_file) as tf:
                for test_file_line in tf:
                    test_file_line = test_file_line.strip()
                    if test_file_line.startswith(("//", "#")) or test_file_line == "":
                        continue
                    matches = re.findall(symbols_regex, test_file_line)
                    for m in matches:
                        if m != "":
                            matched_symbols.add(m)
    return matched_symbols


def parse_symbols(filename: str) -> list[Symbol]:
    """Read header_only_apis.txt, attributing each symbol to the header(s)
    named by the most recent header comment above it."""
    entries: list[Symbol] = []
    headers: list[str] = []
    with open(filename) as f:
        for idx, line in enumerate(f):
            symbol = line.strip()
            if not symbol:
                continue
            if symbol[0] == "#":
                match = HEADER_COMMENT.match(symbol)
                if match:
                    headers = [h.strip() for h in match.group(1).split(",")]
                continue
            entries.append(Symbol(symbol, idx + 1, headers))
    return entries


def is_reachable(symbol: Symbol, pattern: re.Pattern[str]) -> bool:
    """Whether the symbol turns up in any header it is filed under, or in
    anything those headers include."""
    for filed_under in symbol.headers:
        header = resolve_header(filed_under)
        if header is None:
            continue
        if any(pattern.search(_read(f)) for f in include_closure(header)):
            return True
    return False


def check_symbols_exist(filename: str, entries: list[Symbol]) -> list[LintMessage]:
    """Verify each listed symbol is still reachable from the header it is
    filed under, so removing an API cannot leave a stale entry behind."""
    lint_messages: list[LintMessage] = []

    for symbol in entries:
        if IDENTIFIER.match(symbol.name):
            pattern = re.compile(rf"\b{re.escape(symbol.name)}\b")
        else:
            # operators are spelled "operator<<" in the header
            pattern = re.compile(rf"operator\s*{re.escape(symbol.name)}")

        if not symbol.headers:
            description = (
                f"{symbol.name} is not filed under any header. Add a comment "
                "naming its header (e.g. '# torch/headeronly/util/Half.h') "
                "above it so its continued existence can be verified."
            )
        elif is_reachable(symbol, pattern):
            continue
        else:
            description = (
                f"{symbol.name} is listed as a header-only API but was not "
                f"found in {' or '.join(symbol.headers)} (or anything they "
                "include). If it was renamed or removed, update this file to "
                "match -- a stale entry advertises an API that no longer "
                "exists. If it moved, correct the header comment above it."
            )

        lint_messages.append(
            LintMessage(
                path=filename,
                line=symbol.lineno,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[missing-symbol]",
                original=None,
                replacement=None,
                description=description,
            )
        )

    return lint_messages


def check_file(
    filename: str, test_globs: list[str] = CPP_TEST_GLOBS
) -> list[LintMessage]:
    """
    Goes through the header_only_apis.txt file and verifies that all symbols
    within the file can be found tested in an appropriately independent .cpp
    file.

    Note that we expect CPP_TEST_GLOBS to be passed in as test_globs--the
    only reason this is an argument at all is for ease of testing.
    """
    lint_messages: list[LintMessage] = []

    entries = parse_symbols(filename)
    # symbols can in fact be duplicated and come from different headers.
    # we are aware this is a flaw in using simple string matching.
    symbols: dict[str, int] = {e.name: e.lineno for e in entries}

    lint_messages += check_symbols_exist(filename, entries)

    # Why reverse the keys? To allow superset regexes to get matched first in
    # find_matched_symbols. For example, we want Float8_e5m2fnuz to match
    # before Float8_e5m2. Otherwise, both Float8_e5m2fnuz and Float8_e5m2 will
    # match Float8_e5m2
    symbols_regex = re.compile("|".join(sorted(symbols.keys(), reverse=True)))
    matched_symbols = find_matched_symbols(symbols_regex, test_globs)

    for s, lineno in symbols.items():
        if s not in matched_symbols:
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=lineno,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[untested-symbol]",
                    original=None,
                    replacement=None,
                    description=(
                        f"{s} has been included as a header-only API "
                        "but is not tested in any of CPP_TEST_GLOBS, which "
                        f"contains {CPP_TEST_GLOBS}.\n"
                        "Please add a .cpp test using the symbol without "
                        "linking anything to verify that the symbol is in "
                        "fact header-only. If you already have a test but it's"
                        " not found, please add the .cpp file to CPP_TEST_GLOBS"
                        " in tools/linters/adapters/header_only_linter.py."
                    ),
                )
            )

    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="header only APIs linter",
        fromfile_prefix_chars="@",
    )
    args = parser.parse_args()

    for lint_message in check_file(
        str(REPO_ROOT) + "/torch/header_only_apis.txt", CPP_TEST_GLOBS
    ):
        print(json.dumps(lint_message._asdict()), flush=True)
