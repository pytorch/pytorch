from __future__ import annotations

import token
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence
    from tokenize import TokenInfo


__all__ = (
    "Block",
    "EMPTY_TOKENS",
    "FileLinter",
    "LineWithSets",
    "LintResult",
    "ParseError",
    "PythonFile",
    "ROOT",
)

NO_TOKEN = -1

# Python 3.12 and up have two new token types, FSTRING_START and FSTRING_END
_START_OF_LINE_TOKENS = token.DEDENT, token.INDENT, token.NEWLINE
_IGNORED_TOKENS = token.COMMENT, token.ENDMARKER, token.ENCODING, token.NL
EMPTY_TOKENS = dict.fromkeys(_START_OF_LINE_TOKENS + _IGNORED_TOKENS)

_LINTER = Path(__file__).absolute().parents[0]
ROOT = _LINTER.parents[3]


class ParseError(ValueError):
    def __init__(self, token: TokenInfo, *args: str) -> None:
        super().__init__(*args)
        self.token = token


from .block import Block
from .file_linter import FileLinter
from .messages import LintResult
from .python_file import PythonFile
