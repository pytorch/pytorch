from __future__ import annotations

import dataclasses as dc
from enum import Enum


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


@dc.dataclass
class LintMessage:
    """This is a datatype representation of the JSON that gets sent to lintrunner
    as described here:
    https://docs.rs/lintrunner/latest/lintrunner/lint_message/struct.LintMessage.html
    """

    code: str
    name: str
    severity: LintSeverity

    char: int | None = None
    description: str | None = None
    line: int | None = None
    original: str | None = None
    path: str | None = None
    replacement: str | None = None

    asdict = dc.asdict


@dc.dataclass
class LintResult:
    """LintResult is a single result from a linter.

    Like LintMessage but the .length member allows you to make specific edits to
    one location within a file, not just replace the whole file.

    Linters can generate recursive results - results that contain other results.

    For example, the annotation linter would find two results in this code sample:

        index = Union[Optional[str], int]

    And the first result, `Union[Optional[str], int]`, contains the second one,
    `Optional[str]`, so the first result is recursive but the second is not.

    If --fix is selected, the linter does a cycle of tokenizing and fixing all
    the non-recursive edits until no edits remain.
    """

    name: str

    line: int | None = None
    char: int | None = None
    replacement: str | None = None
    length: int | None = None  # Not in LintMessage
    description: str | None = None
    original: str | None = None

    is_recursive: bool = False  # Not in LintMessage

    @property
    def is_edit(self) -> bool:
        return None not in (self.char, self.length, self.line, self.replacement)

    def apply(self, lines: list[str]) -> None:
        if not (
            self.char is None
            or self.length is None
            or self.line is None
            or self.replacement is None
            or self.is_recursive
        ):
            line = lines[self.line - 1]
            before = line[: self.char]
            after = line[self.char + self.length :]
            lines[self.line - 1] = f"{before}{self.replacement}{after}"

    def contains(self, r: LintResult) -> bool:
        assert self.char is not None and self.line is not None
        assert r.char is not None and r.line is not None
        return self.line == r.line and self.char <= r.char and self.end >= r.end

    @property
    def end(self) -> int:
        assert self.char is not None and self.length is not None
        return self.char + self.length

    def as_message(self, code: str, path: str) -> LintMessage:
        d = dc.asdict(self)
        d.pop("is_recursive")
        d.pop("length")
        if self.is_edit:
            # This is one of our , which we don't want to
            # send to lintrunner as a replacement
            d["replacement"] = None

        return LintMessage(code=code, path=path, severity=LintSeverity.ERROR, **d)

    def sort_key(self) -> tuple[int, int, str]:
        line = -1 if self.line is None else self.line
        char = -1 if self.char is None else self.char
        return line, char, self.name
