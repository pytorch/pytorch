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
from typing import Any, Callable, Iterator, Sequence
from typing_extensions import Never


EMPTY_TOKENS = {
    token.COMMENT,
    token.DEDENT,
    token.ENCODING,
    token.INDENT,
    token.NEWLINE,
    token.NL,
}


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

    def as_json(self) -> str:
        return json.dumps(dc.asdict(self), sort_keys=True)


@dc.dataclass
class LintResult:
    """LintResult is a single result from a linter.

    Like LintMessage but the .length member allows you to make specific edits to
    one location within a file, not just replace the whole file."""

    name: str

    char: int | None = None
    description: str | None = None
    length: int | None = None
    line: int | None = None
    original: str | None = None
    replacement: str | None = None

    def apply(self, lines: list[str]) -> bool:
        if (
            self.char is None
            or self.line is None
            or self.length is None
            or self.replacement is None
        ):
            return False

        if ret := None not in (self.char, self.length, self.line, self.replacement):
            line = lines[self.line - 1]
            before = line[: self.char]
            after = line[self.char + self.length :]
            lines[self.line - 1] = f"{before}{self.replacement}{after}"
        return ret

    def as_json(self, code: str, path: str) -> str:
        d = dc.asdict(self)
        d.pop("length")
        msg = LintMessage(code=code, path=path, severity=LintSeverity.ERROR, **d)
        return msg.as_json()

    def sort_key(self) -> tuple[int, int, str]:
        line = -1 if self.line is None else self.line
        char = -1 if self.char is None else self.char
        return line, char, self.name

    def is_edit(self) -> bool:
        return None not in (self.char, self.length, self.line, self.replacement)


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
        is_formatter: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(prog, usage, description, None, **kwargs)
        self._epilog = epilog

        help = "A list of files or directories to lint"
        self.add_argument("files", nargs="*", help=help)
        # , fromfile_prefix_chars="@", type=argparse.FileType

        help = "Fix lint errors if possible" if is_formatter else argparse.SUPPRESS
        self.add_argument("-f", "--fix", action="store_true", help=help)

        help = "Run for lintrunner and print LintMessages"
        self.add_argument("-l", "--lintrunner", action="store_true", help=help)

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
    omitted: OmittedLines
    path: Path | None

    def __init__(
        self, linter_name: str, path: Path | None = None, contents: str | None = None
    ) -> None:
        self.path = path
        if contents is None and path is not None:
            contents = path.read_text()

        self.contents = contents or ""
        self.lines = self.contents.splitlines(keepends=True)
        self.omitted = OmittedLines(self.lines, linter_name)

    @cached_property
    def tokens(self) -> tuple[TokenInfo, ...]:
        # Might raise IndentationError if the code is mal-indented
        return tuple(generate_tokens(iter(self.lines).__next__))


@dc.dataclass
class LintResultsForFile:
    python_file: PythonFile
    results: list[LintResult]

    def __post_init__(self) -> None:
        self.results.sort(key=LintResult.sort_key)

    @property
    def path(self) -> Path:
        assert self.python_file.path
        return self.python_file.path

    def __bool__(self) -> bool:
        return bool(self.results)

    def apply_replacements(self, linter_name: str, is_lintrunner: bool) -> None:
        lines = self.python_file.lines[:]
        for r in reversed(self.results):
            r.apply(lines)

        replacement = "".join(lines)
        if is_lintrunner:
            self.results.append(
                LintResult(
                    name=f"Suggested fixes for {linter_name}",
                    original=self.python_file.contents,
                    replacement=replacement,
                )
            )
        else:
            self.path.write_text(replacement)

    def print_results(self, print: Callable[..., None] = print) -> None:
        lines, path = self.python_file.lines, self.python_file.path
        for r in self.results:
            if r.line is None:
                print(f"{path}: {r.name}")
                continue

            print(f"{path}:{r.line}:{r.char}: {r.name}")
            print()

            begin = max(r.line - ErrorLines.BEFORE, 1)
            end = min(begin + ErrorLines.WINDOW, 1 + len(lines))

            for lineno in range(begin, end):
                source_line = lines[lineno - 1].rstrip()
                print(f"{lineno:5} | {source_line}")
                if lineno == r.line:
                    assert r.char is not None
                    length = getattr(r, "length", 1)
                    print((9 + r.char) * " ", "^" * (length or 1))

            print()


class ErrorLines:
    """How many lines to display before and after an error"""

    WINDOW = 5
    BEFORE = 2
    AFTER = WINDOW - BEFORE - 1


class FileLinter(ABC):
    description: str
    linter_name: str
    is_formatter: bool

    epilog: str | None = None

    @abstractmethod
    def _lint(self, python_file: PythonFile) -> Iterator[LintResult]:
        raise NotImplementedError

    def __init__(self, argv: list[str] | None = None) -> None:
        self.argv = argv
        self.parser = ArgumentParser(
            is_formatter=self.is_formatter,
            description=self.description,
            epilog=self.epilog,
        )

    def lint_all(self, print: Callable[..., None] = print) -> None:
        code = self.linter_name.upper()
        if self.args.fix and self.args.lintrunner:
            raise ValueError("--fix and --lintrunner are incompatible")

        for p in self.paths:
            if file_results := self._lint_file(p):
                file_results.results.sort(key=LintResult.sort_key)
                if self.is_formatter and (self.args.fix or self.args.lintrunner):
                    file_results.apply_replacements(
                        self.linter_name, self.args.lintrunner
                    )

                if self.args.lintrunner:
                    for r in file_results.results:
                        print(r.as_json(code=code, path=str(p)))
                else:
                    file_results.print_results()

    @cached_property
    def args(self) -> Namespace:
        return self.parser.parse_args(self.argv)

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

    def _lint_file(self, p: Path) -> LintResultsForFile:
        if self.args.verbose:
            print(p, "Reading")
        pf = PythonFile(self.linter_name, p)
        try:
            msgs = list(self._lint(pf))
        except IndentationError as e:
            error, (_name, lineno, column, _line) = e.args
            msgs = [LintResult(line=lineno, char=column, name=error)]

        return LintResultsForFile(pf, msgs)

    def set_logging_level(self) -> None:
        if self.args.verbose:
            level = logging.NOTSET
        elif len(self.paths) < 1000:
            level = logging.DEBUG
        else:
            level = logging.INFO

        fmt = "<%(threadName)s:%(levelname)s> %(message)s"
        logging.basicConfig(format=fmt, level=level, stream=sys.stderr)
