from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Sequence, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT not in _PATH:
    from . import _linter
else:
    import _linter

is_name, is_op = _linter.is_name, _linter.is_op

if TYPE_CHECKING:
    from tokenize import TokenInfo


FROM_FUTURE = "from __future__ import annotations"

CONTAINERS = "Dict", "List", "FrozenSet", "Set", "Tuple"
COMBINERS = "Optional", "Union"
DEPRECATED = CONTAINERS + COMBINERS

DESCRIPTION = f"""
`annotations_linter` finds and fixes use old style typing annotations

Old-style requires imports:

    from typing import {", ".join(DEPRECATED)}

    one: Dict[str, int]
    two: Optional[List[one]]
    three: Union[two, Set[str], Tuple[int, ...]]

New:

    {FROM_FUTURE}

    one: dict[str, int]
    two: one | None
    three: two | set[str] | tuple[int, ...]

"""


class AnnotationsLinter(_linter.FileLinter):
    linter_name = "annotations_linter"
    description = DESCRIPTION

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        if not any(line.startswith(FROM_FUTURE) for line in pf.lines[:20]):
            for lineno, line in enumerate(pf.lines):
                if not line.startswith("#"):
                    break
            else:
                return
            yield _linter.LintResult(
                name=f"Add '{FROM_FUTURE}' before this line",
                line=lineno,
                char=0,
                replacement=FROM_FUTURE + "\n",
                length=0,
            )

        token_line: list[TokenInfo]  # mypy falsely complains otherwise.

        for token_line in pf.token_lines:
            if token_line and not (is_name(token_line[0], "from", "imports")):
                for i, t in enumerate(token_line):
                    if is_name(t, *DEPRECATED):
                        yield _lint_token_line(token_line, i)


def _lint_token_line(tokens: Sequence[TokenInfo], i: int) -> _linter.LintResult:
    start = end = i

    def is_multiline() -> bool:
        return tokens[start].start[0] != tokens[end].end[0]

    if i >= 2:
        if is_name(tokens[i - 2], "typing") and is_op(tokens[i - 1], "."):
            start = i - 2

    replacement = None
    t = tokens[i]
    if right_bracket := _close_bracket(tokens, i + 1):
        is_recursive = _contains_deprecated(tokens, i + 1, right_bracket)
    else:
        is_recursive = False

    if t.string in CONTAINERS:
        replacement = t.string.lower()
        name = f"Use '{replacement}' instead of 'typing.{t.string}'"

    else:
        assert t.string in COMBINERS
        if is_optional := (t.string == "Optional"):
            name = "Use '| None' instead of 'typing.Optional'"
        else:
            name = "Use '|' instead of 'typing.Union'"

        if right_bracket:
            end = right_bracket
            if not is_multiline():
                segment = tokens[i + 2 : end]
                if is_optional:
                    s = segment[0]
                    replacement = t.line[s.start[1] : s.end[1]] + " | None"
                else:
                    types = (i.strip() for i in _split_on_commas(segment))
                    replacement = " | ".join(types)

    (line1, char1), (line2, char2) = tokens[start].start, tokens[end].end
    if replacement is None or line1 != line2:
        length = None
    else:
        length = char2 - char1

    return _linter.LintResult(
        name, line1, char1, replacement, length, is_recursive=is_recursive
    )


def _split_on_commas(tokens: Sequence[TokenInfo]) -> Iterator[str]:
    bracket_level = 0
    char1 = tokens[0].start[1]

    for t in tokens:
        if is_op(t):
            bracket_level += (t.string in "[{(") - (t.string in "]})")
            if not bracket_level and t.string == ",":
                char2 = t.start[1]
                yield t.line[char1:char2]
                char1 = char2 + 1

    char2 = t.end[1]
    yield t.line[char1:char2]


def _close_bracket(tokens: Sequence[TokenInfo], i: int) -> int | None:
    if not is_op(tokens[i], "["):
        return None

    bracket_level = 1
    for j in range(i + 1, len(tokens)):
        t = tokens[j]
        if is_op(t, "[", "]"):
            bracket_level += 1 if t.string == "[" else -1
            if not bracket_level:
                return j
    return None


def _contains_deprecated(tokens: Sequence[TokenInfo], begin: int, end: int) -> bool:
    begin = max(begin, 0)
    end = min(end, len(tokens))
    return any(is_name(tokens[i], *DEPRECATED) for i in range(begin, end))


if __name__ == "__main__":
    AnnotationsLinter.run()
