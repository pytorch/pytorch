import dataclasses as dc
from token import NAME, OP
from tokenize import TokenInfo
from typing import Iterable, List


@dc.dataclass
class TokenLine:
    """A logical line of tokens separated by token.NEWLINE.
    There might be physical newlines in it, separated by token.NL.
    """

    tokens: List[TokenInfo] = dc.field(default_factory=list)

    def append(self, t: TokenInfo) -> None:
        self.tokens.append(t)

    def __repr__(self) -> str:
        return " ".join(t.string for t in self.tokens)

    def is_token_using_set(self, i: int) -> bool:
        # This method has to be on the full line, because we look behind and ahead.
        # This is where the logic to recognize `set` goes, and # probably most bug-fixes.

        # Logic to detect sets with { would go first, if this is possible.

        s = i and self.tokens[i - 1]
        t = self.tokens[i]
        u = i < len(self.tokens) - 1 and self.tokens[i + 1]
        if t.string == "Set" and t.type == NAME:
            return u and u.string == "[" and u.type == OP
        if not (t.string == "set" and t.type == NAME):
            return False
        if s and s.string in ("def", "."):
            return False
        return not (u and u.string == "=" and u.type == OP)

    def matching_tokens(self) -> Iterable[TokenInfo]:
        """Matches tokens which use the built-in set"""
        for i, t in enumerate(self.tokens):
            if self.is_token_using_set(i):
                yield t

    def lines_covered(self) -> List[int]:
        lines = sorted(i for t in self.tokens for i in (t.start[0], t.end[0]))
        return list(range(lines[0], lines[-1] + 1)) if lines else []
