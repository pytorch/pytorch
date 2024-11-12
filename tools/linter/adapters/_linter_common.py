from __future__ import annotations

import argparse
import dataclasses as dc
import json
import logging
import sys
import token
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


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


@dc.dataclass
class LintMessage:
    path: str | None = None
    line: int | None = None
    char: int | None = None
    code: str = ""
    severity: LintSeverity = LintSeverity.ERROR
    name: str = ""
    original: str | None = None
    replacement: str | None = None
    description: str | None = None

    asdict = dc.asdict

    def asjson(self, **kwargs: Any) -> str:
        return json.dumps(self.asdict(), **kwargs)


class ParseError(ValueError):
    def __init__(self, token: TokenInfo, *args: str) -> None:
        super().__init__(*args)
        self.token = token

    @classmethod
    def check(cls, cond: Any, token: TokenInfo, *args: str) -> None:
        if not cond:
            raise cls(token, *args)


class ArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        prog: str | None = None,
        usage: str | None = None,
        description: str | None = None,
        epilog: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(prog, usage, description, None, **kwargs)
        self._epilog = epilog
        self.add_argument(
            "files", nargs="*", help="Files or directories to search for sets"
        )
        self.add_argument(
            "-v", "--verbose", action="store_true", help="Print more debug info"
        )

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
    tokens: list[TokenInfo]

    def __init__(self, source: str | Path, linter_name: str) -> None:
        if isinstance(source, Path):
            self.contents = source.read_text()
            self.path = source
        else:
            self.contents = source
            self.path = None
        self.lines = self.contents.splitlines(keepends=True)
        self.tokens = list(generate_tokens(iter(self.lines).__next__))
        self.omitted = OmittedLines(self.lines, linter_name)


class FileLinter:
    linter_name: str  # Assign this in derived classes

    def __init__(self, argv: list[str] | None, *args: str, **kwargs: str) -> None:
        self.argv = argv
        self.parser = ArgumentParser(*args, **kwargs)

    @cached_property
    def args(self) -> Namespace:
        return self.parser.parse_args(self.argv)

    def set_logging_level(self) -> None:
        if self.args.verbose:
            level = logging.NOTSET
        elif len(self.paths) < 1000:
            level = logging.DEBUG
        else:
            level = logging.INFO

        fmt = "<%(threadName)s:%(levelname)s> %(message)s"
        logging.basicConfig(format=fmt, level=level, stream=sys.stderr)

    @cached_property
    def paths(self) -> tuple[Path, ...]:
        files = []
        for fp in self.args.files:
            for f in fp.split(":"):
                if f.startswith("@"):
                    files.extend(Path(f[1:]).read_text().splitlines())
                elif f != "--":
                    files.append(f)
        return tuple(Path(f) for f in files)

    def _lint(self, python_file: PythonFile) -> Iterator[LintMessage]:
        raise NotImplementedError

    def lint_all(self) -> Iterator[LintMessage]:
        for p in self.paths:
            if self.args.verbose:
                print(p, "Reading")
            try:
                pf = PythonFile(p, self.linter_name)
            except IndentationError as e:
                msg, (_name, lineno, column, _line) = e.args
                msgs = [LintMessage(line=lineno, char=column, name="Indentation Error")]
            else:
                msgs = list(self._lint(pf))

            for m in msgs:
                m.code = self.linter_name.upper()
                m.path = str(p)
                yield m

    def print_all(self) -> None:
        for m in self.lint_all():
            print(m.asjson())
