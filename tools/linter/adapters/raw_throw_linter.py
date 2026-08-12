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
_WORD = re.compile(r"[A-Za-z]")
# The marker has to be the whole point of its comment. Prose that merely
# mentions it - as the docs for this linter do - must not license anything.
_MARKER_LINE = re.compile(
    rf"^\s*(?://+!?|/\*+!?|\*)\s*{re.escape(MARKER)}\b(?P<rest>.*)"
)


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
                if _is_continued(text, nl):
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
            end = None
            if text[prefix_start:i] in _RAW_STRING_PREFIXES:
                delimiter = _raw_string_delimiter(text, i)
                if delimiter is not None:
                    terminator = ")" + delimiter + '"'
                    close = text.find(terminator, i + len(delimiter) + 2)
                    if close != -1:
                        end = close + len(terminator)
            if end is None:
                # Not a well-formed raw string, or one that is never
                # terminated. Fall back to the line-bounded scan so a malformed
                # literal cannot blank the rest of the file.
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


def _raw_string_delimiter(text: str, quote: int) -> str | None:
    """The d-char sequence of the raw string opening at `quote`, or None if this
    is not a well-formed raw string. The standard caps the delimiter at 16
    characters and excludes whitespace, parentheses and backslash."""
    for j in range(quote + 1, min(quote + 18, len(text))):
        if text[j] == "(":
            return text[quote + 1 : j]
        if text[j] in ' ()\\"\n\r\t\v\f':
            return None
    return None


def _is_continued(text: str, newline: int) -> bool:
    """Whether the newline at `newline` is a backslash line continuation."""
    i = newline - 1
    if i >= 0 and text[i] == "\r":
        i -= 1
    return i >= 0 and text[i] == "\\"


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
        seen = False
        i = match.end()
        while i < n:
            ch = code[i]
            if not ch.isspace():
                seen = True
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch in ";," and depth == 0:
                break
            elif ch == "\n" and depth == 0 and seen and not _is_continued(code, i):
                # A macro body has no trailing `;`, so without this the scan
                # would run to the end of the file. Requiring `seen` keeps an
                # operand written on the next line attached to its throw, so
                # that it cannot be mistaken for a bare `throw;`, and a
                # clang-format-wrapped throw is inside brackets here anyway.
                break
            i += 1
        # Line continuations inside a macro body are noise, not part of the
        # expression.
        tokens = (t.rstrip("\\") for t in code[match.end() : i].split())
        expression = " ".join(t for t in tokens if t)
        throws.append(Throw(code.count("\n", 0, match.start()) + 1, expression))
    return throws


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
    if not expression:
        return (
            "`throw;` re-raises the exception being handled. If the try/catch "
            f"cannot be restructured away, put `// {MARKER}: <reason>` on the "
            "line immediately above it."
        )
    macro = replacement_macro(path)
    if len(expression) > 80:
        expression = expression[:77] + "..."
    lines = [
        f"`throw {expression}` is a raw throw. Use {macro} instead.",
    ]
    if macro == "TORCH_CHECK":
        lines.append(
            "c10::Error surfaces as RuntimeError, which is already what "
            "HANDLE_TH_ERRORS does with every std:: exception, so TORCH_CHECK "
            "is usually an exact swap. The type only matters if this throw "
            "escapes through a bare pybind11 binding, which maps "
            "std::invalid_argument to ValueError and std::out_of_range to "
            "IndexError; TORCH_CHECK_VALUE and TORCH_CHECK_INDEX preserve those."
        )
    lines.append(
        f"If this throw is correct and must stay, put `// {MARKER}: <reason>` "
        "on the line immediately above it."
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
    throw_lines = {throw.line for throw in throws}

    # A marker licenses exactly one throw: the first one on the line below it.
    # A marker sharing a line with a throw licenses nothing - otherwise a
    # trailing marker would silently cover the *next* line's throw.
    licensed: set[int] = set()
    messages = []
    code_lines = code.split("\n")
    for lineno, comment in enumerate(comments.split("\n"), 1):
        marker = _MARKER_LINE.search(comment)
        if marker is None:
            continue
        # The marker gets a line to itself, so that it cannot trail a throw (and
        # appear to license the next one) or hide at the end of a line of code.
        # A trailing `\` is a line continuation, not code, so that a marker can
        # sit inside a multi-line macro body - which is the only placement
        # available there.
        alone = not code_lines[lineno - 1].strip().rstrip("\\").strip()
        target = lineno + 1
        if not alone or target not in throw_lines:
            messages.append(
                message(
                    lineno,
                    "orphaned-allow-raw-throw",
                    f"`{MARKER}` does not apply to anything: it must sit on the "
                    "line immediately above the throw it allows. It is not a "
                    "file-level opt-out and it cannot trail the throw itself.",
                )
            )
            continue
        licensed.add(target)
        rest = marker.group("rest").strip()
        if not (rest.startswith(":") and _WORD.search(rest)):
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
        messages.append(
            message(throw.line, "raw-throw", describe(path, throw.expression))
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
