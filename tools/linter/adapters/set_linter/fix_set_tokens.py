from __future__ import annotations

import token
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .python_file import PythonFile


IMPORT_LINE = "from torch.utils._ordered_set import OrderedSet\n"
DEBUG = False


def fix_set_tokens(pf: PythonFile, add_any: bool = False) -> None:
    _fix_tokens(pf, add_any)
    _add_import(pf)


def _fix_tokens(pf: PythonFile, add_any: bool) -> None:
    ordered_set = "OrderedSet[Any]" if add_any else "OrderedSet"

    for t in sorted(pf.set_tokens, reverse=True, key=lambda t: t.start):
        (start_line, start_col), (end_line, end_col) = t.start, t.end
        assert start_line == end_line
        line = pf.lines[start_line - 1]

        a, b, c = line[:start_col], line[start_col:end_col], line[end_col:]
        assert b in ("set", "Set")
        pf.lines[start_line - 1] = f"{a}{ordered_set}{c}"


def _add_import(pf: PythonFile) -> None:
    if not pf.set_tokens:
        return

    froms, comments, imports = [], [], []

    for token_line in pf.token_lines:
        t = token_line[0]
        if t.type == token.INDENT:
            DEBUG and print("INDENT", token_line)
            break
        elif t.type == token.COMMENT:
            DEBUG and print("COMMENT", token_line)
            comments.append(token_line)
        elif t.type == token.NAME and t.string in ("from", "import"):
            DEBUG and print("import", token_line)
            if any(
                i.type == token.NAME and i.string == "OrderedSet" for i in token_line
            ):
                return
            elif t.string == "from":
                froms.append(token_line)
            else:
                imports.append(token_line)
        else:
            DEBUG and print("other", t)

    if section := froms or imports or comments:
        insert_before = section[-1][-1].start[0] + 1
    else:
        insert_before = 0
    pf.lines.insert(insert_before, IMPORT_LINE)
