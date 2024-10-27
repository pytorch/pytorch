from __future__ import annotations
from .exceptions import ParseError

from typing import NamedTuple


COMMENTCHARS = "#;"


class _ParsedLine(NamedTuple):
    lineno: int
    section: str | None
    name: str | None
    value: str | None


def parse_lines(path: str, line_iter: list[str]) -> list[_ParsedLine]:
    result: list[_ParsedLine] = []
    section = None
    for lineno, line in enumerate(line_iter):
        name, data = _parseline(path, line, lineno)
        # new value
        if name is not None and data is not None:
            result.append(_ParsedLine(lineno, section, name, data))
        # new section
        elif name is not None and data is None:
            if not name:
                raise ParseError(path, lineno, "empty section name")
            section = name
            result.append(_ParsedLine(lineno, section, None, None))
        # continuation
        elif name is None and data is not None:
            if not result:
                raise ParseError(path, lineno, "unexpected value continuation")
            last = result.pop()
            if last.name is None:
                raise ParseError(path, lineno, "unexpected value continuation")

            if last.value:
                last = last._replace(value=f"{last.value}\n{data}")
            else:
                last = last._replace(value=data)
            result.append(last)
    return result


def _parseline(path: str, line: str, lineno: int) -> tuple[str | None, str | None]:
    # blank lines
    if iscommentline(line):
        line = ""
    else:
        line = line.rstrip()
    if not line:
        return None, None
    # section
    if line[0] == "[":
        realline = line
        for c in COMMENTCHARS:
            line = line.split(c)[0].rstrip()
        if line[-1] == "]":
            return line[1:-1], None
        return None, realline.strip()
    # value
    elif not line[0].isspace():
        try:
            name, value = line.split("=", 1)
            if ":" in name:
                raise ValueError()
        except ValueError:
            try:
                name, value = line.split(":", 1)
            except ValueError:
                raise ParseError(path, lineno, "unexpected line: %r" % line)
        return name.strip(), value.strip()
    # continuation
    else:
        return None, line.strip()


def iscommentline(line: str) -> bool:
    c = line.lstrip()[:1]
    return c in COMMENTCHARS
