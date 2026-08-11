#!/usr/bin/env python3
"""
RAWTHROW: flags raw `throw` statements in C++ code.

Unlike a plain grep this understands C++ lexical structure: comments and
string/char literal bodies are blanked out before matching, so the word
"throw" in prose or in a message never counts, and a clang-format-wrapped

    throw(
        SomeError(...));

is recognised as a single `throw SomeError(...)` rather than a bare `throw(`.

A throw that is genuinely correct can be kept by putting

    // @allow-raw-throw: <why this throw must stay>

on the line immediately above it. The reason is mandatory, and the marker only
applies to the next line - there is no way to switch the check off for a whole
file.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "RAWTHROW"

MARKER = "@allow-raw-throw"

# pybind11's exception types are that library's own protocol for reaching the
# interpreter. No TORCH_CHECK stands in for them.
PYBIND_NAMESPACES = ("py::", "pybind11::")

# Typed exceptions that are control flow rather than error reporting: each one
# is caught by name in the subsystem that throws it, so a TORCH_CHECK would
# break the code that handles it. Add a name here only with that justification,
# never because converting a site looked awkward. Names are matched exactly as
# written at the throw, so a type thrown both qualified and unqualified needs
# both spellings.
ALLOWED_EXCEPTION_TYPES = frozenset(
    {
        "PythonError",  # torch/csrc/fx/node.cpp, caught in the same file
        "WorkerException",  # torch/csrc/api dataloader, carries a worker's error
        "c10::AcceleratorError",  # carries the device error code alongside the message
    }
)


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


class Throw(NamedTuple):
    line: int
    expression: str


# Prefixes that turn a `"` into the start of a raw string literal.
_RAW_STRING_PREFIXES = frozenset({"R", "LR", "uR", "UR", "u8R"})

_IDENT_CHARS = re.compile(r"\w")
_NUMBER_CHARS = re.compile(r"[\w']")
_THROW = re.compile(r"\bthrow\b")
_QUALIFIED_NAME = re.compile(r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*")
_BARE_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_MARKER_LINE = re.compile(rf"{re.escape(MARKER)}(?P<rest>.*)")


def scan_source(text: str) -> tuple[str, str]:
    """Split `text` into a code view and a comment view.

    Both views have the same length and the same newlines as `text`, so an
    offset means the same thing in all three. `code` has comment and literal
    text replaced by spaces; `comments` has everything *but* comment text
    replaced by spaces.
    """
    n = len(text)
    code = list(text)
    comments = [c if c == "\n" else " " for c in text]

    def blank(target: list[str], start: int, end: int) -> None:
        for k in range(start, end):
            if target[k] != "\n":
                target[k] = " "

    def keep(start: int, end: int) -> None:
        comments[start:end] = list(text[start:end])
        blank(code, start, end)

    i = 0
    while i < n:
        c = text[i]

        if c == "/" and text.startswith("//", i):
            # A `//` comment runs to the end of the line, unless the line ends
            # with a backslash, in which case it continues onto the next one.
            end = i
            while True:
                nl = text.find("\n", end)
                if nl == -1:
                    end = n
                    break
                cont = nl - 1
                if cont >= 0 and text[cont] == "\r":
                    cont -= 1
                if cont >= 0 and text[cont] == "\\":
                    end = nl + 1
                    continue
                end = nl
                break
            keep(i, end)
            i = end
            continue

        if c == "/" and text.startswith("/*", i):
            close = text.find("*/", i + 2)
            end = n if close == -1 else close + 2
            keep(i, end)
            i = end
            continue

        if c == '"':
            prefix_start = i
            while prefix_start > 0 and _IDENT_CHARS.match(text[prefix_start - 1]):
                prefix_start -= 1
            if text[prefix_start:i] in _RAW_STRING_PREFIXES:
                open_paren = text.find("(", i + 1)
                if open_paren != -1:
                    terminator = ")" + text[i + 1 : open_paren] + '"'
                    close = text.find(terminator, open_paren + 1)
                    end = n if close == -1 else close + len(terminator)
                    blank(code, i, end)
                    i = end
                    continue
            end = _end_of_quoted(text, i, '"')
            blank(code, i, end)
            i = end
            continue

        if c == "'":
            # A quote glued to the end of a number is a C++14 digit separator
            # (1'000'000), not a literal. A quote glued to an encoding prefix
            # (L'a', u8'a') is a literal.
            token_start = i
            while token_start > 0 and _NUMBER_CHARS.match(text[token_start - 1]):
                token_start -= 1
            if text[token_start:i][:1].isdigit():
                i += 1
                continue
            end = _end_of_quoted(text, i, "'")
            blank(code, i, end)
            i = end
            continue

        i += 1

    return "".join(code), "".join(comments)


def _end_of_quoted(text: str, start: int, quote: str) -> int:
    """Offset just past the literal opening at `start`, or end of line if the
    literal is unterminated."""
    i = start + 1
    n = len(text)
    while i < n:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        if text[i] == "\n":
            return i
        i += 1
    return n


def find_throws(code: str) -> list[Throw]:
    """Find every `throw` token in `code` (which must already have comments and
    literals blanked out) together with the expression it throws."""
    throws = []
    n = len(code)
    for match in _THROW.finditer(code):
        depth = 0
        i = match.end()
        while i < n:
            ch = code[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch in ";," and depth == 0:
                break
            i += 1
        expression = _unwrap(" ".join(code[match.end() : i].split()))
        throws.append(Throw(code.count("\n", 0, match.start()) + 1, expression))
    return throws


def _unwrap(expression: str) -> str:
    """Drop the parentheses clang-format adds around a wrapped throw operand."""
    while expression.startswith("("):
        depth = 0
        for end, char in enumerate(expression):
            depth += (char == "(") - (char == ")")
            if depth == 0:
                break
        if depth or end != len(expression) - 1:
            break
        expression = expression[1:-1].strip()
    return expression


def is_allowed(expression: str) -> bool:
    """Whether this throw is correct as written and needs no annotation.

    Everything else has to become a TORCH_CHECK, or be added to
    ALLOWED_EXCEPTION_TYPES with a justification, or carry a per-site marker.
    """
    if not expression:
        return True  # bare `throw;` re-raises the exception already in flight
    # `throw std::move(e)` is deliberately not allowed. It reads like a rethrow,
    # but every occurrence in this repo constructs a fresh exception and moves
    # it to avoid a copy, so the type still has to be judged on its merits.
    thrown = expression.removeprefix("::")
    if thrown.startswith(PYBIND_NAMESPACES):
        return True
    match = _QUALIFIED_NAME.match(thrown)
    if match is None or match.group(0) not in ALLOWED_EXCEPTION_TYPES:
        return False
    # An allowed name still has to be a constructor call, so that a variable
    # that happens to share the name is not allowed along with it.
    return thrown[match.end() :].lstrip().startswith(("(", "{"))


def display(expression: str) -> str:
    """The thrown expression, for a diagnostic. Arguments are elided because by
    this point string literals in them have been blanked out."""
    prefix = "::" if expression.startswith("::") else ""
    thrown = expression.removeprefix("::")
    match = _QUALIFIED_NAME.match(thrown)
    if match and "(" in thrown:
        return f"{prefix}{match.group(0)}(...)"
    return expression


def replacement_macro(path: str) -> str:
    """The error-reporting macro available to code in `path`. Not everything can
    depend on c10."""
    posix = path.replace("\\", "/")
    if "torch/csrc/inductor/aoti_runtime/" in posix:
        return "AOTI_RUNTIME_CHECK"
    if "torch/headeronly/" in posix or "torch/csrc/stable/" in posix:
        return "STD_TORCH_CHECK"
    return "TORCH_CHECK"


def describe(path: str, expression: str) -> str:
    if _BARE_IDENTIFIER.fullmatch(expression):
        return (
            f"`throw {expression};` throws a copy of `{expression}` with its "
            "static type, so a derived exception would be sliced. Write "
            "`throw;` to re-raise the original."
        )
    macro = replacement_macro(path)
    lines = [
        f"`throw {display(expression)}` is a raw throw. Use {macro} instead.",
    ]
    if macro == "TORCH_CHECK":
        lines.append(
            "Preserve the exception type: std::runtime_error -> TORCH_CHECK, "
            "std::invalid_argument -> TORCH_CHECK_VALUE, "
            "std::out_of_range -> TORCH_CHECK_INDEX. "
            "See c10/util/Exception.h for the full list."
        )
    lines.append(
        "If this exception is typed control flow that something catches by "
        "name, add it to ALLOWED_EXCEPTION_TYPES in "
        "tools/linter/adapters/raw_throw_linter.py with a reason. If this one "
        f"site is correct, put `// {MARKER}: <reason>` on the line immediately "
        "above it."
    )
    return " ".join(lines)


def check_source(path: str, text: str) -> list[LintMessage]:
    code, comments = scan_source(text)

    def message(line: int, name: str, description: str) -> LintMessage:
        return LintMessage(
            path=path,
            line=line,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name=name,
            original=None,
            replacement=None,
            description=description,
        )

    throws = find_throws(code)
    first_on_line: dict[int, Throw] = {}
    for throw in throws:
        first_on_line.setdefault(throw.line, throw)

    # A marker licenses exactly one throw: the first one on the line below it.
    # A marker sharing a line with a throw licenses nothing - otherwise a
    # trailing marker would silently cover the *next* line's throw.
    licensed: set[int] = set()
    messages = []
    for lineno, comment in enumerate(comments.split("\n"), 1):
        marker = _MARKER_LINE.search(comment)
        if marker is None:
            continue
        target = first_on_line.get(lineno + 1)
        if lineno in first_on_line or target is None:
            messages.append(
                message(
                    lineno,
                    "orphaned-allow-raw-throw",
                    f"`{MARKER}` does not apply to anything: it must sit on the "
                    "line immediately above the throw it allows. It is not a "
                    "file-level opt-out and it cannot trail the throw itself.",
                )
            )
        elif is_allowed(target.expression):
            messages.append(
                message(
                    lineno,
                    "redundant-allow-raw-throw",
                    f"`throw {display(target.expression)}` is allowed already, "
                    f"so this `{MARKER}` suppresses nothing. Remove it.",
                )
            )
        else:
            licensed.add(lineno + 1)
            rest = marker.group("rest").strip()
            if not (rest.startswith(":") and rest[1:].strip()):
                messages.append(
                    message(
                        lineno,
                        "allow-raw-throw-without-reason",
                        f"`{MARKER}` must state why the throw is correct, as "
                        f"`// {MARKER}: <reason>`.",
                    )
                )

    for throw in throws:
        if throw.line in licensed:
            licensed.discard(throw.line)
            continue
        if is_allowed(throw.expression):
            continue
        messages.append(
            message(throw.line, "raw throw statement", describe(path, throw.expression))
        )

    messages.sort(key=lambda m: (m.line or 0, m.name))
    return messages


def check_file(path: str) -> list[LintMessage]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as err:
        return [
            LintMessage(
                path=path,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="file-access-error",
                original=None,
                replacement=None,
                description=f"Failed to read file: {err}",
            )
        ]
    if "throw" not in text:
        return []
    return check_source(path, text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="raw throw linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    for filename in args.filenames:
        for lint_message in check_file(filename):
            print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
