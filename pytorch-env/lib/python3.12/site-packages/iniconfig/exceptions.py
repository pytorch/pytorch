from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Final


class ParseError(Exception):
    path: Final[str]
    lineno: Final[int]
    msg: Final[str]

    def __init__(self, path: str, lineno: int, msg: str) -> None:
        super().__init__(path, lineno, msg)
        self.path = path
        self.lineno = lineno
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno + 1}: {self.msg}"
