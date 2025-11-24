from __future__ import annotations

import token
from functools import cached_property
from pathlib import Path
from tokenize import generate_tokens, TokenInfo
from typing import TYPE_CHECKING
from typing_extensions import Self

from . import EMPTY_TOKENS, NO_TOKEN, ParseError, ROOT
from .blocks import blocks
from .sets import LineWithSets


if TYPE_CHECKING:
    from collections.abc import Sequence

    from .block import Block


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
        self.path = path.relative_to(ROOT) if path and path.is_absolute() else path
        self._contents = contents

    @cached_property
    def contents(self) -> str:
        if self._contents is not None:
            return self._contents
        elif self.path is not None:
            return self.path.read_text()
        else:
            return ""

    @cached_property
    def filename(self) -> str:
        return str(self.path)

    @cached_property
    def lines(self) -> list[str]:
        return self.contents.splitlines(keepends=True)

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

    def __getitem__(self, i: int | slice) -> TokenInfo | Sequence[TokenInfo]:
        return self.tokens[i]

    def next_token(self, start: int, token_type: int, error: str) -> int:
        for i in range(start, len(self.tokens)):
            if self.tokens[i].type == token_type:
                return i
        raise ParseError(self.tokens[-1], error)

    def docstring(self, start: int) -> str:
        for i in range(start + 1, len(self.tokens)):
            tk = self.tokens[i]
            if tk.type == token.STRING:
                return tk.string
            if tk.type not in EMPTY_TOKENS:
                return ""
        return ""

    @cached_property
    def indent_to_dedent(self) -> dict[int, int]:
        dedents = dict[int, int]()
        stack = list[int]()

        for i, t in enumerate(self.tokens):
            if t.type == token.INDENT:
                stack.append(i)
            elif t.type == token.DEDENT:
                dedents[stack.pop()] = i

        return dedents

    @cached_property
    def errors(self) -> dict[str, str]:
        return {}

    @cached_property
    def braced_sets(self) -> list[Sequence[TokenInfo]]:
        lines = [t for tl in self._lines_with_sets for t in tl.braced_sets]
        return [s for s in lines if not self.omitted(s)]

    @cached_property
    def sets(self) -> list[TokenInfo]:
        tokens = [t for tl in self._lines_with_sets for t in tl.sets]
        return [t for t in tokens if not self.omitted([t])]

    @cached_property
    def insert_import_line(self) -> int | None:
        froms, imports = self.import_lines
        for i in froms + imports:
            tl = self.token_lines[i]
            if any(i.type == token.NAME and i.string == "OrderedSet" for i in tl):
                return None
        if section := froms or imports:
            return self._lines_with_sets[section[-1]].tokens[-1].start[0] + 1
        return self.opening_comment_lines + 1

    @cached_property
    def _lines_with_sets(self) -> list[LineWithSets]:
        return [LineWithSets(tl) for tl in self.token_lines]

    @cached_property
    def blocks(self) -> list[Block]:
        res = blocks(self.tokens)
        self.errors.update(res.errors)
        return res.blocks

    @cached_property
    def blocks_by_line_number(self) -> dict[int, Block]:
        # Lines that don't appear are in the top-level scope
        # Later blocks correctly overwrite earlier, parent blocks.
        return {i: b for b in self.blocks for i in b.line_range}

    def block_name(self, line: int) -> str:
        block = self.blocks_by_line_number.get(line)
        return block.full_name if block else ""

    @cached_property
    def python_parts(self) -> tuple[str, ...]:
        assert self.path
        parts = self.path.with_suffix("").parts
        return parts[:-1] if parts[-1] == "__init__" else parts


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


if __name__ == "__main__":
    import sys

    for file in sys.argv[1:]:
        print(file)
        pbf = PythonFile("", Path(file)).blocks_by_line_number
        print(*pbf)
