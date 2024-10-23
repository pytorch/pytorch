import token
from pathlib import Path
from tokenize import TokenInfo, tokenize
from typing import List

from .token_line import TokenLine


OMIT_COMMENT = "# noqa: set_linter"

"""
Python's tokenizer splits Python code into lexical tokens tagged with one of many
token names. We are only interested in a few of these: references to the built-in `set`
will have to be in a NAME token, and we're only care about enough context to see if it's a
really `set` or, say, a method `set`.
"""


class PythonFile:
    path: Path
    lines: List[str]
    tokens: List[TokenInfo]
    token_lines: List[TokenLine]
    set_tokens: List[TokenInfo]

    def __init__(self, path: Path) -> None:
        self.path = path
        with self.path.open() as fp:
            self.lines = fp.readlines()

        with self.path.open("rb") as fp:
            self.tokens = list(tokenize(fp.readline))

        self.token_lines = [TokenLine()]
        for t in self.tokens:
            self.token_lines[-1].append(t)
            if t.type == token.NEWLINE:
                self.token_lines.append(TokenLine())

        omitted = OmittedLines(path)
        lines = [tl for tl in self.token_lines if not omitted(tl)]
        self.set_tokens = [t for tl in lines for t in tl.matching_tokens()]


class OmittedLines:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines = set()
        with self.path.open() as fp:
            for i, s in enumerate(fp):
                if s.rstrip().endswith(OMIT_COMMENT):
                    self.lines.add(i + 1)  # Tokenizer lines start at 1

    def __call__(self, tl: TokenLine) -> bool:
        # A TokenLine might span multiple physical lines
        return bool(self.lines.intersection(tl.lines_covered()))
