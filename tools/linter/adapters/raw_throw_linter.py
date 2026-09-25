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
file. A type that is always correct to throw belongs in ALLOWED_EXCEPTION_TYPES
instead, and a marker over one of those is itself reported.
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

# Exceptions that no TORCH_CHECK can stand in for, mapped to the path prefix
# they are allowed under, so an entry cannot leak into code its justification
# does not cover. An empty prefix means the justification holds everywhere. Add
# an entry only with a justification of the kind below, never because converting
# a site looked awkward.
#
# Deliberately absent: py::type_error, py::value_error and py::index_error.
# Those are exactly TORCH_CHECK_TYPE, TORCH_CHECK_VALUE and TORCH_CHECK_INDEX -
# same Python type and same message - because the translator registered in
# torch/csrc/Module.cpp maps c10::TypeError and friends onto PyExc_TypeError.
ALLOWED_EXCEPTION_TYPES = {
    # Typed control flow: each is caught by name, so TORCH_CHECK would break
    # the code that handles it.
    "PythonError": "torch/csrc/fx/",
    "WorkerException": "torch/csrc/api/",
    "py::cast_error": "torch/csrc/jit/",  # caught by name in jit/python
    "py::error_already_set": "",  # a Python error is set; rethrowing preserves it
    "MyException": "c10/test/",  # LeftRight_test, caught by EXPECT_THROW
    # Drives the unwinder's own control flow; caught by name in
    # fast_symbolizer.h and unwind.cpp.
    "UnwindError": "torch/csrc/profiler/",
    "unwind::UnwindError": "torch/csrc/profiler/",
    # Reach the interpreter as a Python type c10 has no equivalent of, so
    # TORCH_CHECK would turn them into RuntimeError.
    "py::key_error": "",
    "py::stop_iteration": "",
    # Carries the device error code alongside the message.
    "c10::AcceleratorError": "",
    # Same shape as AcceleratorError: carries the ncclResult_t alongside the
    # message, and is what this backend's own NCCL_CHECK macros raise.
    "NCCLException": "torch/csrc/distributed/c10d/nccl2/",
}


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
_TYPE_NAME = re.compile(r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*(?:<[^<>]*>)?")
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
        expression = _unwrap(" ".join(t for t in tokens if t))
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
        if depth or end != len(expression) - 1 or not expression[1:-1].strip():
            break
        expression = expression[1:-1].strip()
    return expression


def is_allowed(path: str, expression: str) -> bool:
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
    match = _QUALIFIED_NAME.match(thrown)
    if match is None:
        return False
    allowed_under = ALLOWED_EXCEPTION_TYPES.get(match.group(0))
    if allowed_under is None:
        return False
    # lintrunner passes absolute paths, so match the scope as a run of path
    # segments rather than anchoring at the start of the string.
    posix = "/" + path.replace("\\", "/")
    if f"/{allowed_under}" not in posix:
        return False
    # The constructor call has to be the whole operand. Checking only that a
    # `(` follows would let any violation be laundered by prefixing an allowed
    # one, as in `throw py::key_error(m).with_context(x)`.
    return _is_whole_call(thrown[match.end() :].lstrip())


def _as_statement(expression: str) -> str:
    """The throw written back out as source, for a diagnostic."""
    return f"throw {display(expression)};" if expression else "throw;"


def _is_whole_call(rest: str) -> bool:
    """Whether `rest` is exactly one balanced `(...)` or `{...}` and nothing
    else."""
    if not rest or rest[0] not in "({":
        return False
    depth = 0
    for i, char in enumerate(rest):
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
            if depth == 0:
                return not rest[i + 1 :].strip()
    return False


def display(expression: str) -> str:
    """The thrown expression, for a diagnostic. Constructor arguments are elided
    because by this point string literals in them have been blanked out."""
    prefix = "::" if expression.startswith("::") else ""
    thrown = expression.removeprefix("::")
    match = _TYPE_NAME.match(thrown)
    if match:
        rest = thrown[match.end() :].lstrip()
        for opening, closing in (("(", ")"), ("{", "}")):
            if rest.startswith(opening):
                return f"{prefix}{match.group(0)}{opening}...{closing}"
    if len(expression) > 80:
        return expression[:77] + "..."
    return expression


def replacement_macro(path: str) -> str:
    """The error-reporting macro available to code in `path`. Not everything can
    depend on c10."""
    posix = path.replace("\\", "/")
    if "torch/csrc/inductor/aoti_runtime/" in posix:
        return "AOTI_RUNTIME_CHECK"
    if (
        "torch/headeronly/" in posix
        or "torch/csrc/stable/" in posix
        # Installed, and deliberately depends on torch/headeronly only.
        or posix.endswith("torch/csrc/utils/generated_serialization_types.h")
    ):
        return "STD_TORCH_CHECK"
    return "TORCH_CHECK"


def describe(path: str, expression: str) -> str:
    macro = replacement_macro(path)
    if _BARE_IDENTIFIER.fullmatch(expression):
        lines = [
            f"`throw {expression};` throws a copy of `{expression}` with its "
            "static type, so a derived exception would be sliced. If it is "
            "re-raising the exception being handled, write `throw;`; if it is "
            f"reporting a fresh error, use {macro}.",
        ]
    else:
        lines = [
            f"`throw {display(expression)}` is a raw throw. Use {macro} instead.",
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
    throws_on_line: dict[int, list[Throw]] = {}
    for throw in throws:
        throws_on_line.setdefault(throw.line, []).append(throw)

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
        target = throws_on_line.get(lineno + 1)
        if not alone or target is None:
            messages.append(
                message(
                    lineno,
                    "orphaned-allow-raw-throw",
                    f"`{MARKER}` does not apply to anything: it must sit on the "
                    "line immediately above the throw it allows. It is not a "
                    "file-level opt-out and it cannot trail the throw itself.",
                )
            )
        elif all(is_allowed(path, t.expression) for t in target):
            messages.append(
                message(
                    lineno,
                    "redundant-allow-raw-throw",
                    f"`{_as_statement(target[0].expression)}` is allowed "
                    f"already, so this `{MARKER}` suppresses nothing. Remove it.",
                )
            )
        else:
            licensed.add(lineno + 1)
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
        if is_allowed(path, throw.expression):
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
