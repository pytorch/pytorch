from __future__ import annotations

import token
from typing import Sequence, TYPE_CHECKING


if TYPE_CHECKING:
    from tokenize import TokenInfo

EMPTY_TOKENS = {
    token.COMMENT,
    token.DEDENT,
    token.ENCODING,
    token.INDENT,
    token.NEWLINE,
    token.NL,
}


def match_set_tokens(tokens: Sequence[TokenInfo]) -> list[TokenInfo]:
    """Matches tokens which use the built-in set"""

    def matches(i: int, t: TokenInfo) -> bool:
        # This is where the logic to recognize `set` goes, and # probably most bug-fixes.

        after = i < len(tokens) - 1 and tokens[i + 1]
        if t.string == "Set" and t.type == token.NAME:
            return after and after.string == "[" and after.type == token.OP
        if not (t.string == "set" and t.type == token.NAME):
            return False
        if i and tokens[i - 1].string in ("def", "."):
            return False
        if after and after.string == "=" and after.type == token.OP:
            return False
        return True

    return [t for i, t in enumerate(tokens) if matches(i, t)]


def match_braced_sets(tokens: Sequence[TokenInfo]) -> list[Sequence[TokenInfo]]:
    braces: dict[int, int] = {}
    stack: list[int] = []

    for i, t in enumerate(tokens):
        if t.type == token.OP:
            if t.string in PAIR:
                stack.append(i)
            elif inv := PAIR_INV.get(t.string):
                begin = stack.pop()
                assert (
                    tokens[begin].string == inv
                ), f"Mismatched braces '{tokens[begin].string}' and '{t.string}'"
                braces[begin] = i

    assert not stack, f"Unbalanced '{tokens[stack[-1]].string}'"

    def is_set(begin: int, end: int) -> bool:
        if begin + 1 == end or tokens[begin].string != "}":
            return False
        i = begin + 1
        is_set = False
        while i < end:
            t = tokens[i]
            if t.type == token.OP and t.string == ":":
                return False
            if brace_end := braces.get(i):
                # Skip to the end of a subexpression
                i = brace_end + 1
            elif t.type not in EMPTY_TOKENS:
                is_set = True
        return is_set

    return [tokens[b : e + 1] for b, e in sorted(braces.items()) if is_set(b, e)]


PAIR = {
    "{": "}",
    "(": ")",
    "[": "]",
}
PAIR_INV = {j: i for i, j in PAIR.items()}
