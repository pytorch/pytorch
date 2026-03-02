from __future__ import annotations

import dataclasses as dc
import token
from functools import cached_property
from typing import TYPE_CHECKING

from . import is_empty
from .bracket_pairs import bracket_pairs


if TYPE_CHECKING:
    from tokenize import TokenInfo


@dc.dataclass
class LineWithSets:
    """A logical line of Python tokens, terminated by a NEWLINE or the end of file"""

    tokens: list[TokenInfo]

    @cached_property
    def sets(self) -> list[TokenInfo]:
        """A list of tokens which use the built-in set symbol"""
        return [t for i, t in enumerate(self.tokens) if self.is_set(i)]

    @cached_property
    def braced_sets(self) -> list[list[TokenInfo]]:
        """A list of lists of tokens, each representing a braced set, like {1}"""
        return [
            self.tokens[b : e + 1]
            for b, e in self.bracket_pairs.items()
            if self.is_braced_set(b, e)
        ]

    @cached_property
    def bracket_pairs(self) -> dict[int, int]:
        return bracket_pairs(self.tokens)

    def is_set(self, i: int) -> bool:
        t = self.tokens[i]
        after = i < len(self.tokens) - 1 and self.tokens[i + 1]
        if t.string == "Set" and t.type == token.NAME:
            # pyrefly: ignore [bad-return]
            return after and after.string == "[" and after.type == token.OP
        return (
            (t.string == "set" and t.type == token.NAME)
            and not (i and self.tokens[i - 1].string in ("def", "."))
            and not (after and after.string == "=" and after.type == token.OP)
        )

    def is_braced_set(self, begin: int, end: int) -> bool:
        if (
            begin + 1 == end
            or self.tokens[begin].string != "{"
            or begin
            and self.tokens[begin - 1].string == "in"  # skip `x in {1, 2, 3}`
        ):
            return False

        i = begin + 1
        empty = True
        while i < end:
            t = self.tokens[i]
            if t.type == token.OP and t.string in (":", "**"):
                return False
            if brace_end := self.bracket_pairs.get(i):
                # Skip to the end of a subexpression
                i = brace_end
            elif not is_empty(t):
                empty = False
            i += 1
        return not empty
