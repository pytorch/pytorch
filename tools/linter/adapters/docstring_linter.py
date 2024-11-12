from __future__ import annotations

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
    from tokenize import TokenInfo


MAX_LINES = {"class": 100, "def": 80}

MIN_DOCSTRING = 16  # docstrings shorter than this are ignored
IGNORE_PROTECTED = True  # If True, ignore classes and files whose names start with _.

ERROR_FMT = "Every {type} with more than {length} lines needs a docstring"

DESCRIPTION = """`docstring_linter` reports on long functions, methods or classes
without docstrings"""


def _is_def(t: TokenInfo) -> bool:
    return t.type == token.NAME and t.string in ("class", "def")


class DocstringLinter(FileLinter):
    linter_name = "docstring_linter"

    def __init__(self, argv: list[str] | None = None) -> None:
        super().__init__(argv, description=DESCRIPTION)

        help = "Maximum number of lines for an undocumented class"
        self.parser.add_argument(
            "--max-class", "-c", default=MAX_LINES["class"], type=int, help=help
        )

        help = "Maximum number of lines for an undocumented function"
        self.parser.add_argument(
            "--max-function", "-f", default=MAX_LINES["def"], type=int, help=help
        )

        help = "Minimum number of characters for a docstring"
        self.parser.add_argument(
            "--min-docstring", "-d", default=MIN_DOCSTRING, type=int, help=help
        )

        help = "Lint functions, methods and classes that start with _"
        self.parser.add_argument(
            "--lint-protected", "-p", action="store_true", help=help
        )

    @cached_property
    def max_lines(self) -> dict[str, int]:
        return {"class": self.args.max_class, "def": self.args.max_function}

    def _lint(self, pf: PythonFile) -> Iterator[LintMessage]:
        tokens = pf.tokens
        indents = indent_to_dedent(tokens)
        defs = [i for i, t in enumerate(tokens) if _is_def(t)]

        def next_token(start: int, token_type: int, error: str) -> int:  # type: ignore[return]
            for i in range(start, len(tokens)):
                if (t := tokens[i]).type == token_type:
                    return i
            ParseError.check(False, tokens[-1], error)

        for i in defs:
            name = next_token(i + 1, token.NAME, "Definition with no name")
            if not self.args.lint_protected and tokens[name].string.startswith("_"):
                continue

            indent = next_token(name + 1, token.INDENT, "Definition with no indent")
            dedent = indents[indent]

            lines = tokens[dedent].start[0] - tokens[indent].start[0]
            max_lines = self.max_lines[tokens[i].string]
            if lines <= max_lines:
                continue

            # Now search for a docstring
            docstring_len = -1
            for k in range(indent + 1, len(tokens)):
                tk = tokens[k]
                if tk.type == token.STRING:
                    docstring_len = len(tk.string)
                    break
                if tk.type not in EMPTY_TOKENS:
                    break

            if docstring_len >= self.args.min_docstring:
                continue

            # Now check if it's omitted
            if pf.omitted(pf.tokens[i:indent]):
                continue

            t = tokens[i]
            def_name = "function" if t.string == "def" else t.string
            msg = f"docstring found for {def_name} '{tokens[name].string}'"
            if docstring_len < 0:
                msg = "No " + msg
            else:
                msg = msg + " was too short"
            yield LintMessage(line=t.start[0], char=t.start[1], name=msg)


def indent_to_dedent(tokens: Sequence[TokenInfo]) -> dict[int, int]:
    indent_to_dedent: dict[int, int] = {}
    stack: list[int] = []

    for i, t in enumerate(tokens):
        if t.type == token.INDENT:
            stack.append(i)
        elif t.type == token.DEDENT:
            assert stack
            indent_to_dedent[stack.pop()] = i

    assert not stack
    return indent_to_dedent


if __name__ == "__main__":
    DocstringLinter().print_all()
