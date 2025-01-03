from __future__ import annotations

import dataclasses as dc
import sys
import token
from functools import cached_property
from pathlib import Path
from typing import Iterator, Sequence, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from tokenize import TokenInfo


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


class SetLinter(_linter.FileLinter):
    linter_name = "set_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        pl = PythonLines(pf)
        for b in pl.braced_sets:
            yield _linter.LintResult(ERROR, *b[0].start, "OrderedSet([", 1)
            yield _linter.LintResult(ERROR, *b[-1].start, "])", 1)

        for b in pl.sets:
            yield _linter.LintResult(ERROR, *b.start, "OrderedSet", 3)

        if (pl.sets or pl.braced_sets) and (ins := pl.insert_import_line) is not None:
            yield _linter.LintResult(
                "Add import for OrderedSet", ins, 0, IMPORT_LINE, 0
            )


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
        return _linter.bracket_pairs(self.tokens)

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
            elif t.type not in _linter.EMPTY_TOKENS:
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

    def __init__(self, pf: _linter.PythonFile) -> None:
        self.contents = pf.contents
        self.lines = pf.lines
        self.path = pf.path
        self.tokens = pf.tokens
        self.omitted = pf.omitted

        self.token_lines = [TokenLine(tl) for tl in pf.token_lines]

        sets = [t for tl in self.token_lines for t in tl.sets]
        self.sets = [s for s in sets if not pf.omitted([s])]

        braced_sets = [t for tl in self.token_lines for t in tl.braced_sets]
        self.braced_sets = [s for s in braced_sets if not pf.omitted(s)]

        froms, imports = pf.import_lines
        for i in froms + imports:
            tl = pf.token_lines[i]
            if any(i.type == token.NAME and i.string == "OrderedSet" for i in tl):
                self.insert_import_line = None
                return

        if section := froms or imports:
            self.insert_import_line = pf.token_lines[section[-1]][-1].start[0] + 1
        else:
            self.insert_import_line = 0


if __name__ == "__main__":
    SetLinter.run()
