from __future__ import annotations

import dataclasses as dc
import token
from functools import cached_property
from typing import Iterator, Sequence, TYPE_CHECKING

from ._linter_common import (
    EMPTY_TOKENS,
    FileLinter,
    LintMessage,
    ParseError,
    PythonFile,
)


if TYPE_CHECKING:
    from pathlib import Path
    from tokenize import TokenInfo


BRACKETS = {"{": "}", "(": ")", "[": "]"}
BRACKETS_INV = {j: i for i, j in BRACKETS.items()}
ERROR = "Builtin `set` is deprecated"
IMPORT_LINE = "from torch.utils._ordered_set import OrderedSet\n"

DESCRIPTION = """`set_linter` is a lintrunner linter which finds usages of the
Python built-in class `set` in Python code, and optionally replaces them with
`OrderedSet`.
"""

EPILOG = """
`lintrunner` operates on whole commits. If you want to remove uses of `set`
from existing files not part of a commit, call `set_linter` directly:

    python tools/linter/adapters/set_linter.py --fix [... python files ...]

---

To omit a line of Python code from `set_linter` checking, append a comment:

    s = set()  # noqa: set_linter
    t = {  # noqa: set_linter
       "one",
       "two",
    }

---

Running set_linter in fix mode (though either `lintrunner -a` or `--fix`
should not significantly change the behavior of working code, but will still
usually needs some manual intervention:

1. Replacing `set` with `OrderedSet` will sometimes introduce new typechecking
errors because `OrderedSet` is imperfectly generic. Find a common type for its
elements (in the worst case, `typing.Any` always works), and use
`OrderedSet[YourCommonTypeHere]`.

2. The fix mode doesn't recognize generator expressions, so it replaces:

    s = {i for i in range(3)}

with

    s = OrderedSet([i for i in range(3)])

You can and should delete the square brackets in every such case.

3. There is a common pattern of set usage where a set is created and then only
used for testing inclusion. For small collections, up to around 12 elements, a
tuple is more time-efficient than an OrderedSet and also has less visual clutter
(see https://github.com/rec/test/blob/master/python/time_access.py).
"""


class SetLinter(FileLinter):
    linter_name = "set_linter"

    def __init__(self, argv: list[str] | None = None) -> None:
        super().__init__(argv, description=DESCRIPTION, epilog=EPILOG)

        help = "Fix the files in the repository directly without lintrunner"
        self.parser.add_argument("-f", "--fix", action="store_true", help=help)

    def _lint(self, pf: PythonFile) -> Iterator[LintMessage]:
        pl = PythonLines(pf)
        if not (replacements := sorted(lint_replacements(pl), reverse=True)):
            return

        lines = pl.lines[:]
        messages: list[LintMessage] = []

        for r in replacements:
            messages.append(LintMessage(line=r.line, char=r.char, name=r.name))

            line = lines[r.line - 1]
            before, after = line[: r.char], line[r.char + r.length :]
            lines[r.line - 1] = f"{before}{r.replacement}{after}"

        rep = "".join(lines)
        if not self.args.fix:
            yield from messages

            name = "Suggested fixes for set_linter"
            yield LintMessage(name=name, original=pl.contents, replacement=rep)

        elif pf.path:
            pf.path.write_text(rep)
            print("Rewrote", pf.path)


@dc.dataclass(order=True)
class LintReplacement:
    line: int
    char: int
    length: int
    replacement: str
    name: str = ERROR


def lint_replacements(pl: PythonLines) -> Iterator[LintReplacement]:
    for b in pl.braced_sets:
        yield LintReplacement(*b[0].start, 1, "OrderedSet([")
        yield LintReplacement(*b[-1].start, 1, "])")

    for b in pl.sets:
        yield LintReplacement(*b.start, 3, "OrderedSet")

    if (pl.sets or pl.braced_sets) and (ins := pl.insert_import_line()) is not None:
        yield LintReplacement(ins, 0, 0, IMPORT_LINE, "Add import for OrderedSet")


@dc.dataclass
class TokenLine:
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
        braces: dict[int, int] = {}
        stack: list[int] = []

        for i, t in enumerate(self.tokens):
            if t.type == token.OP:
                if t.string in BRACKETS:
                    stack.append(i)
                elif inv := BRACKETS_INV.get(t.string):
                    ParseError.check(stack, t, "Never opened")
                    begin = stack.pop()
                    braces[begin] = i

                    b = self.tokens[begin].string
                    ParseError.check(b == inv, t, f"Mismatched braces '{b}' at {begin}")

        ParseError.check(not stack, t, "Left open")
        return braces

    def is_set(self, i: int) -> bool:
        t = self.tokens[i]
        after = i < len(self.tokens) - 1 and self.tokens[i + 1]
        if t.string == "Set" and t.type == token.NAME:
            return after and after.string == "[" and after.type == token.OP
        if not (t.string == "set" and t.type == token.NAME):
            return False
        if i and self.tokens[i - 1].string in ("def", "."):
            return False
        if after and after.string == "=" and after.type == token.OP:
            return False
        return True

    def is_braced_set(self, begin: int, end: int) -> bool:
        if begin + 1 == end or self.tokens[begin].string != "{":
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
            elif t.type not in EMPTY_TOKENS:
                empty = False
            i += 1
        return not empty


class PythonLines:
    """A list of lines of Python code represented by strings"""

    braced_sets: list[Sequence[TokenInfo]]
    contents: str
    lines: list[str]
    path: Path | None
    sets: list[TokenInfo]
    token_lines: list[TokenLine]
    tokens: list[TokenInfo]

    def __init__(self, pf: PythonFile) -> None:
        self.contents = pf.contents
        self.lines = pf.lines
        self.path = pf.path
        self.tokens = pf.tokens
        self.omitted = pf.omitted

        self.token_lines = list(self._split_into_token_lines())

        sets = [t for tl in self.token_lines for t in tl.sets]
        self.sets = [t for t in sets if not self.omitted([t])]

        braced_sets = [t for tl in self.token_lines for t in tl.braced_sets]
        self.braced_sets = [t for t in braced_sets if not self.omitted(t)]

    def _split_into_token_lines(self) -> Iterator[TokenLine]:
        token_line = TokenLine([])

        for t in self.tokens:
            if t.type != token.ENDMARKER:
                token_line.tokens.append(t)
                if t.type == token.NEWLINE:
                    yield token_line
                    token_line = TokenLine([])

        if token_line.tokens:
            yield token_line

    def insert_import_line(self) -> int | None:
        froms, imports = [], []

        for token_line in self.token_lines:
            tokens = token_line.tokens
            tokens = [t for t in tokens if t.type not in (token.COMMENT, token.NL)]
            t = tokens[0]
            if t.type == token.INDENT:
                break
            if not (t.type == token.NAME and t.string in ("from", "import")):
                continue
            if any(i.type == token.NAME and i.string == "OrderedSet" for i in tokens):
                return None
            if t.string == "from":
                froms.append(tokens)
            else:
                imports.append(tokens)

        if section := froms or imports:
            return section[-1][-1].start[0] + 1
        return 0


if __name__ == "__main__":
    SetLinter().print_all()
