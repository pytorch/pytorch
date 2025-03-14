from __future__ import annotations

import argparse
import dataclasses as dc
import json
import logging
import sys
import token
from abc import ABC, abstractmethod
from argparse import Namespace
from enum import Enum
from functools import cached_property
from pathlib import Path
from tokenize import generate_tokens
from typing import Any, Generic, get_args, TYPE_CHECKING
from typing_extensions import Never, Self, TypeVar


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from tokenize import TokenInfo


__all__ = (
    "Block",
    "bracket_pairs",
    "EMPTY_TOKENS",
    "FileLinter",
    "LineWithSets",
    "LintResult",
    "ParseError",
    "PythonFile",
    "ROOT",
)

# Python 3.12 and up have two new token types, FSTRING_START and FSTRING_END
NO_TOKEN = -1

START_OF_LINE_TOKENS = dict.fromkeys((token.DEDENT, token.INDENT, token.NEWLINE))
IGNORED_TOKENS = dict.fromkeys(
    (token.COMMENT, token.ENDMARKER, token.ENCODING, token.NL)
)
EMPTY_TOKENS = START_OF_LINE_TOKENS | IGNORED_TOKENS

LINTER = Path(__file__).absolute().parents[0]
ROOT = LINTER.parents[3]


class ParseError(ValueError):
    def __init__(self, token: TokenInfo, *args: str) -> None:
        super().__init__(*args)
        self.token = token


from .block import Block
from .bracket_pairs import bracket_pairs
from .file_linter import FileLinter
from .messages import LintResult
from .python_file import PythonFile
from .sets import LineWithSets
