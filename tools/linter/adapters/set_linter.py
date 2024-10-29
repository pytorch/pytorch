from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
import token
from argparse import Namespace
from functools import cached_property
from pathlib import Path
from tokenize import generate_tokens, TokenInfo
from typing import Any, Iterator, Sequence


IMPORT_LINE = "from torch.utils._ordered_set import OrderedSet\n"
DEBUG = False

ERROR = "Builtin `set` is deprecated"

CONFIG_FILE = "setlint.json"
EMPTY_TOKENS = {
    token.COMMENT,
    token.DEDENT,
    token.ENCODING,
    token.INDENT,
    token.NEWLINE,
    token.NL,
}

BRACKETS = {
    "{": "}",
    "(": ")",
    "[": "]",
}
BRACKETS_INV = {j: i for i, j in BRACKETS.items()}

OMIT = "# noqa: set_linter"


def main() -> None:
    args = get_args(CONFIG_FILE)
    if not (python_files := resolve_python_files(args.files, args.exclude)):
        sys.exit("No files selected")

    for f in python_files:
        error_code = lint_file(f, args)
        sys.exit(error_code)


def lint_file(path: Path, args: Namespace) -> int:
    def source_line(lineno: Any, text: Any = None) -> None:
        if text is None:
            lineno = text = ""
        print(f"{lineno:5} | {text.rstrip()}")

    if args.verbose:
        print(path, "Reading")

    pl = PythonLines(path.read_text())
    if not pl.sets or pl.braced_sets:
        if args.verbose:
            print(path, "OK")
        return 0

    if not args.fix:
        return len(pl.sets) + len(pl.braced_sets)

    fix_set_tokens(pl, args.add_any)
    with path.open("w") as fp:
        fp.writelines(pl.lines)

    count = len(pl.sets)
    print(f"{path}: Fixed {count} error{'s' * (count != 1)}")

    if not args.verbose:
        return len(pl.braced_sets)

    padded = [None] * ErrorLines.BEFORE + pl.lines + [None] * ErrorLines.AFTER
    padded_line = list(enumerate(padded))

    errors = pl.sets + [b[0] for b in pl.braced_sets]
    errors.sort(key=lambda t: t.start)
    for t in errors:
        print()
        (line, start), (line2, end) = t.start, t.end

        window = padded_line[line - 1 : line - 1 + ErrorLines.WINDOW]
        before, after = window[: ErrorLines.BEFORE + 1], window[ErrorLines.BEFORE + 1 :]

        print(f"{path}:{line}:{start}: {ERROR}")

        for j, text in before:
            source_line(j + line - ErrorLines.BEFORE, text)

        source_line("", " " * start + "^" * len(t.string) + "\n")

        for j, text in after:
            source_line(j + line - ErrorLines.BEFORE, text)
        return len(pl.braced_sets)


class ErrorLines:
    """How many lines to display before and after an error"""

    WINDOW = 5
    BEFORE = 2
    AFTER = WINDOW - BEFORE - 1


class ParseError(ValueError):
    def __init__(self, token: TokenInfo, *args: str) -> None:
        super().__init__(*args)
        self.token = token


@dc.dataclass
class TokenLine:
    """A logical line of Python tokens, terminated by a NEWLINE or the end of file"""

    tokens: list[TokenInfo]

    @cached_property
    def sets(self) -> list[TokenInfo]:
        """A list of tokens which use the built-in set symbol"""
        return [t for i, t in enumerate(self.tokens) if self.is_set(i)]

    @cached_property
    def braced_sets(self) -> list[list[TokenInfo]]:
        """A list of lists of tokens, each representing a braced set, like {1}"""
        return [
            self.tokens[b : e + 1]
            for b, e in self.bracket_pairs.items()
            if self.is_braced_set(b, e)
        ]

    @cached_property
    def bracket_pairs(self) -> dict[int, int]:
        braces: dict[int, int] = {}
        stack: list[int] = []

        def check(cond: Any, token: TokenInfo, *args: str) -> None:
            if not cond:
                raise ParseError(token, *args)

        for i, t in enumerate(self.tokens):
            if t.type == token.OP:
                if t.string in BRACKETS:
                    stack.append(i)
                elif inv := BRACKETS_INV.get(t.string):
                    check(stack, t, "Never opened")
                    begin = stack.pop()
                    braces[begin] = i

                    b = self.tokens[begin].string
                    check(b == inv, t, f"Mismatched braces '{b}' at {begin}")

        if self.tokens:
            check(not stack, t, "Left open")
        return braces

    def is_set(self, i: int) -> bool:
        t = self.tokens[i]
        after = i < len(self.tokens) - 1 and self.tokens[i + 1]
        if t.string == "Set" and t.type == token.NAME:
            return after and after.string == "[" and after.type == token.OP
        if not (t.string == "set" and t.type == token.NAME):
            return False
        if i and self.tokens[i - 1].string in ("def", "."):
            return False
        if after and after.string == "=" and after.type == token.OP:
            return False
        return True

    def is_braced_set(self, begin: int, end: int) -> bool:
        if begin + 1 == end or self.tokens[begin].string != "{":
            return False
        i = begin + 1
        is_set = False
        while i < end:
            t = self.tokens[i]
            if t.type == token.OP and t.string == ":":
                return False
            if brace_end := self.bracket_pairs.get(i):
                # Skip to the end of a subexpression
                i = brace_end + 1
            elif t.type not in EMPTY_TOKENS:
                is_set = True
            i += 1
        return is_set


class PythonLines:
    """A list of lines of Python code represented by strings"""

    braced_sets: list[Sequence[TokenInfo]]
    lines: list[str]
    sets: list[TokenInfo]
    token_lines: list[TokenLine]
    tokens: list[TokenInfo]

    def __init__(self, contents: Path | str) -> None:
        text = contents if isinstance(contents, str) else contents.read_text()
        self.lines = text.splitlines(keepends=True)
        self.tokens = list(generate_tokens(iter(self.lines).__next__))
        self.token_lines = list(split_into_token_lines(self.tokens))
        self.omitted = OmittedLines(self.lines)

        sets = [t for tl in self.token_lines for t in tl.sets]
        self.sets = [t for t in sets if not self.omitted([t])]

        braced_sets = [t for tl in self.token_lines for t in tl.braced_sets]
        self.braced_sets = [t for t in braced_sets if not self.omitted(t)]


def split_into_token_lines(tokens: Sequence[TokenInfo]) -> Iterator[TokenLine]:
    token_line = TokenLine([])

    for t in tokens:
        if t.type != token.ENDMARKER:
            token_line.tokens.append(t)
            if t.type == token.NEWLINE:
                yield token_line
                token_line = TokenLine([])

    if token_line.tokens:
        yield token_line


def fix_set_tokens(pl: PythonLines, add_any: bool = False) -> None:
    if pl.sets:
        fix_tokens(pl, add_any)
        add_import(pl)


def fix_tokens(pl: PythonLines, add_any: bool) -> None:
    ordered_set = "OrderedSet[Any]" if add_any else "OrderedSet"

    for t in sorted(pl.sets, reverse=True, key=lambda t: t.start):
        (start_line, start_col), (end_line, end_col) = t.start, t.end
        assert start_line == end_line
        line = pl.lines[start_line - 1]

        a, b, c = line[:start_col], line[start_col:end_col], line[end_col:]
        assert b in ("set", "Set")
        pl.lines[start_line - 1] = f"{a}{ordered_set}{c}"


def add_import(pl: PythonLines) -> None:
    froms, comments, imports = [], [], []

    for token_line in pl.token_lines:
        tokens = token_line.tokens
        t = tokens[0]
        if t.type == token.INDENT:
            DEBUG and print("INDENT", tokens)
            break
        elif t.type == token.COMMENT:
            DEBUG and print("COMMENT", tokens)
            comments.append(tokens)
        elif t.type == token.NAME and t.string in ("from", "import"):
            DEBUG and print("import", tokens)
            if any(i.type == token.NAME and i.string == "OrderedSet" for i in tokens):
                return
            elif t.string == "from":
                froms.append(tokens)
            else:
                imports.append(tokens)
        else:
            DEBUG and print("other", t)

    if section := froms or imports or comments:
        insert_before = section[-1][-1].start[0] + 1
    else:
        insert_before = 0
    pl.lines.insert(insert_before, IMPORT_LINE)


class OmittedLines:
    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = lines
        self.omitted = {i + 1 for i, s in enumerate(lines) if s.rstrip().endswith(OMIT)}

    def __call__(self, tokens: Sequence[TokenInfo]) -> bool:
        # A token_line might span multiple physical lines
        lines = sorted(i for t in tokens for i in (t.start[0], t.end[0]))
        lines_covered = list(range(lines[0], lines[-1] + 1)) if lines else []
        return bool(self.omitted.intersection(lines_covered))


def get_args(config_filename: str) -> argparse.Namespace:
    args = make_parser().parse_args()
    add_configs(args, config_filename)
    return args


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add("files", nargs="*", help="Files or directories to include")
    add("-a", "--add-any", action="store_true", help="Insert OrderedSet[Any]")
    add("-e", "--exclude", action="append", help="Files to exclude from the check")
    add("-f", "--fix", default=None, action="store_true", help="Fix any issues")
    add("-v", "--verbose", default=None, action="store_true", help="Print more info")

    return parser


def add_configs(args: argparse.Namespace, filename: str) -> argparse.Namespace:
    try:
        with open(filename) as fp:
            config = json.load(fp)
    except FileNotFoundError:
        config = {}

    if bad := set(config) - set(vars(args)):
        s = "" if len(bad) == 1 else "s"
        bad_name = ", ".join(sorted(bad))
        raise ValueError(f"Unknown arg{s}: {bad_name}")

    for k, v in config.items():
        if k == "fix" and args.fix is None:
            args.fix = v
        else:
            setattr(args, k, getattr(args, k) or v)

    return args


def resolve_python_files(include: list[str], exclude: list[str]) -> list[Path]:
    include = [j for i in include for j in i.split(":")]
    exclude = [j for i in exclude or () for j in i.split(":")]

    iglobs = python_glob(include, check_errors=True)
    eglobs = python_glob(exclude, check_errors=False)

    return sorted(iglobs - eglobs)


def python_glob(strings: Sequence[str], *, check_errors: bool) -> set[Path]:
    result: set[Path] = set()

    nonexistent: list[str] = []
    not_python: list[str] = []

    for s in strings:
        p = Path(s).expanduser()
        if p.is_dir():
            result.update(p.glob("**/*.py"))
        elif p.suffix != ".py":
            not_python.append(str(p))
        elif p.exists():
            result.add(p)
        else:
            nonexistent.append(str(p))

    if check_errors and (nonexistent or not_python):
        raise ValueError(
            "\n".join(
                [
                    (nonexistent and f'Nonexistent: {" ".join(nonexistent)}') or "",
                    (not_python and f'Not Python: {" ".join(not_python)}') or "",
                ]
            )
        )

    return result


if __name__ == "__main__":
    main()
