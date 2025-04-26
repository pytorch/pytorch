from __future__ import annotations

import token
from functools import cached_property
from pathlib import Path
from tokenize import generate_tokens, TokenInfo
from typing import TYPE_CHECKING
from typing_extensions import Self

from . import NO_TOKEN, ROOT


if TYPE_CHECKING:
    from collections.abc import Sequence


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
        self.path = path and (path.relative_to(ROOT) if path.is_absolute() else path)
        if contents is None and path is not None:
            contents = path.read_text()

        self.contents = contents or ""
        self.lines = self.contents.splitlines(keepends=True)

    @classmethod
    def make(cls, linter_name: str, pc: Path | str | None = None) -> Self:
        if isinstance(pc, Path):
            return cls(linter_name, path=pc)
        return cls(linter_name, contents=pc)

    def with_contents(self, contents: str) -> Self:
        return self.__class__(self.linter_name, self.path, contents)

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

    @cached_property
    def opening_comment_lines(self) -> int:
        """The number of comments at the very top of the file."""
        it = (i for i, s in enumerate(self.lines) if not s.startswith("#"))
        return next(it, 0)


class OmittedLines:
    """Read lines textually and find comment lines that end in 'noqa {linter_name}'"""

    omitted: set[int]

    def __init__(self, lines: Sequence[str], linter_name: str) -> None:
        self.lines = lines
        suffix = f"# noqa: {linter_name}"
        omitted = ((i, s.rstrip()) for i, s in enumerate(lines))
        self.omitted = {i + 1 for i, s in omitted if s.endswith(suffix)}

    def __call__(
        self, tokens: Sequence[TokenInfo], begin: int = 0, end: int = NO_TOKEN
    ) -> bool:
        if end == NO_TOKEN:
            end = len(tokens)
        # A token_line might span multiple physical lines
        start = min((tokens[i].start[0] for i in range(begin, end)), default=0)
        end = max((tokens[i].end[0] for i in range(begin, end)), default=-1)
        return self.contains_lines(start, end)

    def contains_lines(self, begin: int, end: int) -> bool:
        return bool(self.omitted.intersection(range(begin, end + 1)))
