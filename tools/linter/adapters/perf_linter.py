from __future__ import annotations

import enum
import sys
import token
from pathlib import Path
from typing import TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from tokenize import TokenInfo


DESCRIPTION = "`perf_linter` finds common Python patterns which are slow and suggests faster alternatives."
EPILOG = """
You can use this either via `lintrunner` or directly, via:

    python tools/linter/adapters/perf_linter.py --fix [... python files ...]

---

To omit a line of Python code from `perf_linter` checking, append a comment:

    s = ",".join(x for x in range(3))  # noqa: perf_linter

---

perf_linter makes the following suggestions:

1) Generators to list comprehensions:

    >>> timeit.timeit("','.join(str(x) for x in range(10))")
    0.978244581958279
    >>> timeit.timeit("','.join([str(x) for x in range(10)])")
    0.695843324996531

2) List constructors:

    >>> timeit.timeit("list(range(2))")
    0.10411133896559477
    >>> timeit.timeit("[*range(2)]")
    0.08048266009427607

3) Empty list/dict constructors:

    >>> timeit.timeit('list()', number=10000000)
    0.38338504498824477
    >>> timeit.timeit('[]', number=10000000)
    0.18809661595150828
    >>> timeit.timeit('dict()', number=10000000)
    0.3867137918714434
    >>> timeit.timeit('{}', number=10000000)
    0.17520155780948699

"""
ERROR1 = "Generators are slow! Use list comprehensions instead."
ERROR2 = "list(x) is slow! Use [*x] instead."
ERROR3 = "list()/dict() is slow! Use []/{} instead."


class GenExprState(enum.Enum):
    FIRST_TOKEN = enum.auto()
    BEFORE_FOR = enum.auto()
    AFTER_FOR = enum.auto()
    AFTER_IN = enum.auto()
    INVALID = enum.auto()


known_generator_consuming_functions = {
    "list",
    "set",
    "frozenset",
    "dict",
    "tuple",
    "OrderedSet",
    "immutable_list",
    "immutable_dict",
    "min",
    "max",
    "join",
    "sorted",
    "sum",
    "extend",
    "update",
}


def lint_generators(
    tl: list[TokenInfo], brackets: dict[int, int]
) -> Iterator[_linter.LintResult]:
    """Find generator expressions in a token line using a state machine."""
    for start, end in brackets.items():
        if tl[start].string != "(" or not prev_name_in(
            tl, start, known_generator_consuming_functions
        ):
            continue
        state = GenExprState.FIRST_TOKEN
        for tok in iter_tokens_skip_braces(tl, brackets, start, end):
            if tok.string == "for":
                if state != GenExprState.BEFORE_FOR:
                    state = GenExprState.INVALID
                    break
                state = GenExprState.AFTER_FOR
            elif tok.string == ",":
                # , only allowed between for and in
                if state != GenExprState.AFTER_FOR:
                    state = GenExprState.INVALID
                    break
            elif tok.string == "in":
                if state != GenExprState.AFTER_FOR:
                    state = GenExprState.INVALID
                    break
                state = GenExprState.AFTER_IN

            if state == GenExprState.FIRST_TOKEN:
                # `for` should not be the first token
                state = GenExprState.BEFORE_FOR

        if state == GenExprState.AFTER_IN:
            yield _linter.LintResult(ERROR1, *tl[start].start, "([", 1)
            yield _linter.LintResult(ERROR1, *tl[end].start, "])", 1)
            return  # only one lint per line


class ListConstructorState(enum.Enum):
    NO_ARGS = enum.auto()
    HAS_ARG = enum.auto()
    # if there are operators in the argument, we need to do a [*(x+1)] instead of [*x+1]
    HAS_ARG_WITH_OPS = enum.auto()
    INVALID = enum.auto()


def lint_list_constructors(
    tl: list[TokenInfo], brackets: dict[int, int]
) -> Iterator[_linter.LintResult]:
    """Find list(x) using a state machine"""
    for start, end in brackets.items():
        if not (
            start > 0 and tl[start - 1].string == "list" and tl[start].string == "("
        ):
            continue

        state = ListConstructorState.NO_ARGS
        for tok in iter_tokens_skip_braces(tl, brackets, start, end):
            if state == ListConstructorState.NO_ARGS:
                state = ListConstructorState.HAS_ARG
            if (
                tok.type == token.OP
                and tok.string not in ".()[]"
                and state == ListConstructorState.HAS_ARG
            ) or (tok.string in {"and", "or", "not"}):
                state = ListConstructorState.HAS_ARG_WITH_OPS
            if tok.string == ",":
                # list only takes one arg, but mark invalid to be save
                state = ListConstructorState.INVALID
                break

        if state in (
            ListConstructorState.HAS_ARG,
            ListConstructorState.HAS_ARG_WITH_OPS,
        ):
            yield _linter.LintResult(
                ERROR2, *tl[start - 1].start, "", len(tl[start - 1].string)
            )
            if state == ListConstructorState.HAS_ARG:
                yield _linter.LintResult(ERROR2, *tl[start].start, "[*", 1)
                yield _linter.LintResult(ERROR2, *tl[end].start, "]", 1)
            else:
                yield _linter.LintResult(ERROR2, *tl[start].start, "[*(", 1)
                yield _linter.LintResult(ERROR2, *tl[end].start, ")]", 1)
            return  # only one lint per line


def lint_empty_constructors(
    tl: list[TokenInfo], brackets: dict[int, int]
) -> Iterator[_linter.LintResult]:
    """Find dict() instances"""
    for start, end in brackets.items():
        if (
            start > 0
            and start == end - 1
            and tl[start - 1].string in {"dict", "list"}
            and tl[start].string == "("
        ):
            start = start - 1  # "dict" or "list"
            yield _linter.LintResult(
                ERROR3, *tl[start].start, "", len(tl[start].string)
            )
            if tl[start].string == "dict":
                yield _linter.LintResult(ERROR3, *tl[start + 1].start, "{", 1)
                yield _linter.LintResult(ERROR3, *tl[start + 2].start, "}", 1)
            else:
                assert tl[start].string == "list"
                yield _linter.LintResult(ERROR3, *tl[start + 1].start, "[", 1)
                yield _linter.LintResult(ERROR3, *tl[start + 2].start, "]", 1)
            return  # only one lint per line


def iter_tokens_skip_braces(
    tl: list[TokenInfo], brackets: dict[int, int], start: int, end: int
) -> Iterator[TokenInfo]:
    i = start + 1
    while i < end:
        if i in brackets:
            # skip over sub-brackets, but still yield the ) token
            i = brackets[i]
        if tl[i].type != token.COMMENT:
            yield tl[i]
        i += 1


def prev_name_in(tl: list[TokenInfo], i: int, names: set[str]) -> bool:
    while i > 0 and tl[i].type != token.NAME:
        i -= 1
    return tl[i].string in names


class PerfLinter(_linter.FileLinter):
    linter_name = "perf_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        for tl in pf.token_lines:
            brackets = _linter.bracket_pairs(tl)
            for linter in (
                lint_generators,
                lint_list_constructors,
                lint_empty_constructors,
            ):
                lints = [*linter(tl, brackets)]
                if lints:
                    yield from lints
                    break  # only one lint per line


if __name__ == "__main__":
    PerfLinter.run()
