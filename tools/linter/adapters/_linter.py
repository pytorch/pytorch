from __future__ import annotations

import argparse
import dataclasses as dc
import json
import logging
import sys
import token
from abc import ABC, abstractmethod
from argparse import Namespace
from enum import Enum
from functools import cached_property
from pathlib import Path
from tokenize import generate_tokens, TokenInfo
from typing import Any, Iterator, Sequence
from typing_extensions import Never


EMPTY_TOKENS = {
    token.COMMENT,
    token.DEDENT,
    token.ENCODING,
    token.INDENT,
    token.NEWLINE,
    token.NL,
}
BRACKETS = {"{": "}", "(": ")", "[": "]"}
BRACKETS_INV = {j: i for i, j in BRACKETS.items()}


def is_name(t: TokenInfo, *names: str) -> bool:
    return t.type == token.NAME and not names or t.string in names


def is_op(t: TokenInfo, *names: str) -> bool:
    return t.type == token.OP and not names or t.string in names


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


@dc.dataclass
class LintMessage:
    """This is a datatype representation of the JSON that gets sent to lintrunner
    as described here:
    https://docs.rs/lintrunner/latest/lintrunner/lint_message/struct.LintMessage.html
    """

    code: str
    name: str
    severity: LintSeverity

    char: int | None = None
    description: str | None = None
    line: int | None = None
    original: str | None = None
    path: str | None = None
    replacement: str | None = None

    asdict = dc.asdict


@dc.dataclass
class LintResult:
    """LintResult is a single result from a linter.

    Like LintMessage but the .length member allows you to make specific edits to
    one location within a file, not just replace the whole file.

    Linters can generate recursive results - results that contain other results.

    For example, the annotation linter would find two results in this code sample:

        index = Union[Optional[str], int]

    And the first result, `Union[Optional[str], int]`, contains the second one,
    `Optional[str]`, so the first result is recursive but the second is not.

    If --fix is selected, the linter does a cycle of tokenizing and fixing all
    the non-recursive edits until no edits remain.
    """

    name: str

    line: int | None = None
    char: int | None = None
    replacement: str | None = None
    length: int | None = None  # Not in LintMessage
    description: str | None = None
    original: str | None = None

    is_recursive: bool = False  # Not in LintMessage

    @property
    def is_edit(self) -> bool:
        return None not in (self.char, self.length, self.line, self.replacement)

    def apply(self, lines: list[str]) -> bool:
        if self.line is None:
            return False
        line = lines[self.line - 1]

        if self.char is None:
            return False
        before = line[: self.char]

        if self.length is None:
            return False
        after = line[self.char + self.length :]

        lines[self.line - 1] = f"{before}{self.replacement}{after}"
        return True

    def as_message(self, code: str, path: str) -> LintMessage:
        d = dc.asdict(self)
        d.pop("is_recursive")
        d.pop("length")
        if self.is_edit:
            # This is one of our , which we don't want to
            # send to lintrunner as a replacement
            d["replacement"] = None

        return LintMessage(code=code, path=path, severity=LintSeverity.ERROR, **d)

    def sort_key(self) -> tuple[int, int, str]:
        line = -1 if self.line is None else self.line
        char = -1 if self.char is None else self.char
        return line, char, self.name


class ParseError(ValueError):
    def __init__(self, token: TokenInfo, *args: str) -> None:
        super().__init__(*args)
        self.token = token

    @classmethod
    def check(cls, cond: Any, token: TokenInfo, *args: str) -> None:
        if not cond:
            raise cls(token, *args)


class ArgumentParser(argparse.ArgumentParser):
    """
    Adds better help formatting and default arguments to argparse.ArgumentParser
    """

    def __init__(
        self,
        prog: str | None = None,
        usage: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        is_fixer: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(prog, usage, description, None, **kwargs)
        self._epilog = epilog

        help = "A list of files or directories to lint"
        self.add_argument("files", nargs="*", help=help)
        # TODO(rec): get fromfile_prefix_chars="@", type=argparse.FileType to work

        help = "Fix lint errors if possible" if is_fixer else argparse.SUPPRESS
        self.add_argument("-f", "--fix", action="store_true", help=help)

        help = "Run for lintrunner and print LintMessages which aren't edits"
        self.add_argument("-l", "--lintrunner", action="store_true", help=help)

        help = "Run for test, print all LintMessages"
        self.add_argument("-t", "--test", action="store_true", help=help)

        help = "Print more debug info"
        self.add_argument("-v", "--verbose", action="store_true", help=help)

    def exit(self, status: int = 0, message: str | None = None) -> Never:
        """
        Overriding this method is a workaround for argparse throwing away all
        line breaks when printing the `epilog` section of the help message.
        """
        argv = sys.argv[1:]
        if self._epilog and not status and "-h" in argv or "--help" in argv:
            print(self._epilog)
        super().exit(status, message)


class OmittedLines:
    """Read lines textually and find comment lines that end in 'noqa {linter_name}'"""

    omitted: set[int]

    def __init__(self, lines: Sequence[str], linter_name: str) -> None:
        self.lines = lines
        suffix = f"# noqa: {linter_name}"
        omitted = ((i, s.rstrip()) for i, s in enumerate(lines))
        self.omitted = {i + 1 for i, s in omitted if s.endswith(suffix)}

    def __call__(self, tokens: Sequence[TokenInfo]) -> bool:
        # A token_line might span multiple physical lines
        lines = sorted(i for t in tokens for i in (t.start[0], t.end[0]))
        lines_covered = list(range(lines[0], lines[-1] + 1)) if lines else []
        return bool(self.omitted.intersection(lines_covered))


class PythonFile:
    contents: str
    lines: list[str]
    path: Path | None
    linter_name: str

    def __init__(
        self,
        linter_name: str,
        path: Path | None = None,
        contents: str | None = None,
    ) -> None:
        self.linter_name = linter_name
        self.path = path
        if contents is None and path is not None:
            contents = path.read_text()

        self.contents = contents or ""
        self.lines = self.contents.splitlines(keepends=True)

    @classmethod
    def make(cls, linter_name: str, pc: Path | str | None = None) -> PythonFile:
        if isinstance(pc, Path):
            return cls(linter_name, path=pc)
        return cls(linter_name, contents=pc)

    def with_contents(self, contents: str) -> PythonFile:
        return PythonFile(self.linter_name, self.path, contents)

    @cached_property
    def omitted(self) -> OmittedLines:
        assert self.linter_name is not None
        return OmittedLines(self.lines, self.linter_name)

    @cached_property
    def tokens(self) -> list[TokenInfo]:
        # Might raise IndentationError if the code is mal-indented
        return list(generate_tokens(iter(self.lines).__next__))

    @cached_property
    def token_lines(self) -> list[list[TokenInfo]]:
        """Returns lists of TokenInfo segmented by token.NEWLINE"""
        token_lines: list[list[TokenInfo]] = [[]]

        for t in self.tokens:
            if t.type not in (token.COMMENT, token.ENDMARKER, token.NL):
                token_lines[-1].append(t)
                if t.type == token.NEWLINE:
                    token_lines.append([])
        if token_lines and not token_lines[-1]:
            token_lines.pop()
        return token_lines

    @cached_property
    def import_lines(self) -> list[list[int]]:
        froms, imports = [], []
        for i, (t, *_) in enumerate(self.token_lines):
            if t.type == token.INDENT:
                break
            if t.type == token.NAME:
                if t.string == "from":
                    froms.append(i)
                elif t.string == "import":
                    imports.append(i)

        return [froms, imports]


def bracket_pairs(tokens: Sequence[TokenInfo]) -> dict[int, int]:
    """Returns a dictionary mapping opening to closing brackets"""
    braces: dict[int, int] = {}
    stack: list[int] = []

    for i, t in enumerate(tokens):
        if t.type == token.OP:
            if t.string in BRACKETS:
                stack.append(i)
            elif inv := BRACKETS_INV.get(t.string):
                ParseError.check(stack, t, "Never opened")
                begin = stack.pop()
                braces[begin] = i

                b = tokens[begin].string
                ParseError.check(b == inv, t, f"Mismatched braces '{b}' at {begin}")

    if tokens:
        ParseError.check(not stack, t, "Left open")
    return braces


class ErrorLines:
    """How many lines to display before and after an error"""

    WINDOW = 5
    BEFORE = 2
    AFTER = WINDOW - BEFORE - 1


class FileLinter(ABC):
    """The base class that all token-based linters inherit from"""

    description: str
    linter_name: str

    epilog: str | None = None
    is_fixer: bool = True
    report_column_numbers: bool = False

    @abstractmethod
    def _lint(self, python_file: PythonFile) -> Iterator[LintResult]:
        raise NotImplementedError

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv
        self.parser = ArgumentParser(
            is_fixer=self.is_fixer,
            description=self.description,
            epilog=self.epilog,
        )

    @classmethod
    def run(cls) -> Never:
        linter = cls()
        success = linter.lint_all()
        sys.exit(not success)

    def lint_all(self) -> bool:
        if self.args.fix and self.args.lintrunner:
            raise ValueError("--fix and --lintrunner are incompatible")

        success = True
        for p in self.paths:
            success = self._lint_file(p) and success
        return self.args.lintrunner or success

    @cached_property
    def args(self) -> Namespace:
        args = self.parser.parse_args(self.argv)
        args.lintrunner = args.lintrunner or args.test

        return args

    @cached_property
    def code(self) -> str:
        return self.linter_name.upper()

    @cached_property
    def paths(self) -> list[Path]:
        files = []
        file_parts = (f for fp in self.args.files for f in fp.split(":"))
        for f in file_parts:
            if f.startswith("@"):
                files.extend(Path(f[1:]).read_text().splitlines())
            elif f != "--":
                files.append(f)
        return sorted(Path(f) for f in files)

    def _lint_file(self, p: Path) -> bool:
        if self.args.verbose:
            print(p, "Reading")

        pf = PythonFile(self.linter_name, p)
        replacement, results = self._replace(pf)

        print(*self._display(pf, results), sep="\n")
        if results and self.args.fix and pf.path and pf.contents != replacement:
            pf.path.write_text(replacement)

        return not results or self.args.fix and all(r.is_edit for r in results)

    def _replace(self, pf: PythonFile) -> tuple[str, list[LintResult]]:
        # Because of recursive replacements, we need to repeat replacing and reparsing
        # from the inside out until all possible replacements are complete
        previous_result_count = float("inf")
        first_results = None
        original = replacement = pf.contents

        while True:
            try:
                results = list(self._lint(pf))
            except IndentationError as e:
                error, (_name, lineno, column, _line) = e.args
                results = [LintResult(error, lineno, column)]

            if first_results is None:
                first_results = sorted(results, key=LintResult.sort_key)

            if not results or len(results) >= previous_result_count:
                break
            previous_result_count = len(results)

            lines = pf.lines[:]
            for r in reversed(results):
                if not r.is_recursive:
                    r.apply(lines)
            replacement = "".join(lines)

            if not any(r.is_recursive for r in results):
                break
            pf = pf.with_contents(replacement)

        if first_results and self.args.lintrunner:
            name = f"Suggested fixes for {self.linter_name}"
            msg = LintResult(name=name, original=original, replacement=replacement)
            first_results.append(msg)

        return replacement, first_results

    def _display(self, pf: PythonFile, results: list[LintResult]) -> Iterator[str]:
        """Emit a series of human-readable strings representing the results"""
        show_edits = not self.args.fix or self.args.verbose

        first = True
        for r in results:
            if show_edits or r.is_edit:
                if self.args.test or self.args.lintrunner:
                    msg = r.as_message(code=self.code, path=str(pf.path))
                    yield json.dumps(msg.asdict(), sort_keys=True)
                    continue
                if first:
                    first = False
                else:
                    yield ""
                if r.line is None:
                    yield f"{pf.path}: {r.name}"
                else:
                    yield from (i.rstrip() for i in self._display_window(pf, r))

    def _display_window(self, pf: PythonFile, r: LintResult) -> Iterator[str]:
        """Display a window onto the code with an error"""
        if r.char is None or not self.report_column_numbers:
            yield f"{pf.path}:{r.line}: {r.name}"
        else:
            yield f"{pf.path}:{r.line}:{r.char + 1}: {r.name}"

        begin = max((r.line or 0) - ErrorLines.BEFORE, 1)
        end = min(begin + ErrorLines.WINDOW, 1 + len(pf.lines))

        for lineno in range(begin, end):
            source_line = pf.lines[lineno - 1].rstrip()
            yield f"{lineno:5} | {source_line}"
            if lineno == r.line:
                spaces = 8 + (r.char or 0)
                carets = len(source_line) if r.char is None else (r.length or 1)
                yield spaces * " " + carets * "^"


def set_logging_level(args: argparse.Namespace, paths: Sequence[Path | str]) -> None:
    if args.verbose:
        level = logging.NOTSET
    elif len(paths) < 1000:
        level = logging.DEBUG
    else:
        level = logging.INFO

    fmt = "<%(threadName)s:%(levelname)s> %(message)s"
    logging.basicConfig(format=fmt, level=level, stream=sys.stderr)
