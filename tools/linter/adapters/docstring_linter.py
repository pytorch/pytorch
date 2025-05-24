from __future__ import annotations

import dataclasses as dc
import itertools
import json
import sys
import token
from enum import Enum
from functools import cached_property, total_ordering
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING
from typing_extensions import Self


_FILE = Path(__file__).absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _FILE.parent not in _PATH:
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from tokenize import TokenInfo


GRANDFATHER_LIST = Path(str(_FILE).replace(".py", "-grandfather.json"))

# We tolerate a 10% increase in block size before demanding a docstring
TOLERANCE_PERCENT = 10

MAX_LINES = {"class": 100, "def": 80}

MIN_DOCSTRING = 50  # docstrings shorter than this are too short

DESCRIPTION = """
`docstring_linter` reports on long functions, methods or classes without docstrings
""".strip()

METHOD_OVERRIDE_HINT = (
    "If the method overrides a method on a parent class, adding the"
    " `@typing_extensions.override` decorator will make this error"
    " go away."
)


@total_ordering
@dc.dataclass
class Block:
    """A block of Python code starting with either `def` or `class`"""

    class Category(str, Enum):
        CLASS = "class"
        DEF = "def"

    category: Category

    # The sequence of tokens that contains this Block.
    # Tokens are represented in `Block` as indexes into `self.tokens`
    tokens: Sequence[TokenInfo] = dc.field(repr=False)

    # The name of the function or class being defined
    name: str

    # The index of the very first token in the block (the "class" or "def" keyword)
    begin: int

    # The index of the first INDENT token for this block
    indent: int

    # The index of the DEDENT token for this end of this block
    dedent: int

    # The docstring for the block
    docstring: str

    # These next members only get filled in after all blocks have been constructed
    # and figure out family ties

    # The full qualified name of the block within the file.
    # This is the name of this block and all its parents, joined with `.`.
    full_name: str = ""

    # The index of this block within the full list of blocks in the file
    index: int = 0

    # Is this block contained within a function definition?
    is_local: bool = dc.field(default=False, repr=False)

    # Is this block a function definition in a class definition?
    is_method: bool = dc.field(default=False, repr=False)

    # A block index to the parent of this block, or None for a top-level block.
    parent: Optional[int] = None

    # A list of block indexes for the children
    children: list[int] = dc.field(default_factory=list)

    @property
    def start_line(self) -> int:
        return self.tokens[max(self.indent, self.index)].start[0]

    @property
    def end_line(self) -> int:
        return self.tokens[max(self.dedent, self.index)].start[0]

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line

    @property
    def is_class(self) -> bool:
        return self.category == Block.Category.CLASS

    @property
    def display_name(self) -> str:
        """A user-friendly name like 'class One' or 'def One.method()'"""
        ending = "" if self.is_class else "()"
        return f"{self.category.value} {self.full_name}{ending}"

    @cached_property
    def decorators(self) -> list[str]:
        """A list of decorators for this function or method.

        Each decorator both the @ symbol and any arguments to the decorator
        but no extra whitespace.
        """
        return _get_decorators(self.tokens, self.begin)

    @cached_property
    def is_override(self) -> bool:
        return not self.is_class and any(
            d.rpartition(".")[2] == "override" for d in self.decorators
        )

    DATA_FIELDS = (
        "category",
        "children",
        "decorators",
        "display_name",
        "docstring",
        "full_name",
        "index",
        "is_local",
        "is_method",
        "line_count",
        "parent",
        "start_line",
    )

    def as_data(self) -> dict[str, Any]:
        d = {i: getattr(self, i) for i in self.DATA_FIELDS}
        d["category"] = d["category"].value
        return d

    @property
    def is_init(self) -> bool:
        return not self.is_class and self.name == "__init__"

    def contains(self, b: Block) -> bool:
        return self.start_line < b.start_line and self.end_line >= b.end_line

    def __eq__(self, o: object) -> bool:
        assert isinstance(o, Block)
        return o.tokens is self.tokens and o.index == self.index

    def __hash__(self) -> int:
        return super().__hash__()

    def __lt__(self, o: Self) -> bool:
        assert isinstance(o, Block) and o.tokens is self.tokens
        return o.index < self.index


_IGNORE = {token.COMMENT, token.DEDENT, token.INDENT, token.NL}


def _get_decorators(tokens: Sequence[TokenInfo], block_start: int) -> list[str]:
    def decorators() -> Iterator[str]:
        rev = reversed(range(block_start))
        newlines = (i for i in rev if tokens[i].type == token.NEWLINE)
        newlines = itertools.chain(newlines, [-1])  # To account for the first line

        it = iter(newlines)
        end = next(it, -1)  # Like itertools.pairwise in Python 3.10
        for begin in it:
            for i in range(begin + 1, end):
                t = tokens[i]
                if t.type == token.OP and t.string == "@":
                    useful = (t for t in tokens[i:end] if t.type not in _IGNORE)
                    yield "".join(s.string.strip("\n") for s in useful)
                    break
                elif t.type not in _IGNORE:
                    return  # A statement means no more decorators
            end = begin

    out = list(decorators())
    out.reverse()
    return out


class DocstringFile(_linter.PythonFile):
    def __getitem__(self, i: int | slice) -> TokenInfo | Sequence[TokenInfo]:
        return self.tokens[i]

    def next_token(self, start: int, token_type: int, error: str) -> int:
        for i in range(start, len(self.tokens)):
            if self.tokens[i].type == token_type:
                return i
        raise _linter.ParseError(self.tokens[-1], error)

    def docstring(self, start: int) -> str:
        for i in range(start + 1, len(self.tokens)):
            tk = self.tokens[i]
            if tk.type == token.STRING:
                return tk.string
            if tk.type not in _linter.EMPTY_TOKENS:
                return ""
        return ""

    @cached_property
    def indent_to_dedent(self) -> dict[int, int]:
        dedents = dict[int, int]()
        stack = list[int]()

        for i, t in enumerate(self.tokens):
            if t.type == token.INDENT:
                stack.append(i)
            elif t.type == token.DEDENT:
                dedents[stack.pop()] = i

        return dedents

    @cached_property
    def errors(self) -> dict[str, str]:
        return {}

    @cached_property
    def blocks(self) -> list[Block]:
        blocks: list[Block] = []

        for i in range(len(self.tokens)):
            try:
                if (b := self.block(i)) is not None:
                    blocks.append(b)
            except _linter.ParseError as e:
                self.errors[e.token.line] = " ".join(e.args)

        for i, parent in enumerate(blocks):
            for j in range(i + 1, len(blocks)):
                if parent.contains(child := blocks[j]):
                    child.parent = i
                    parent.children.append(j)
                else:
                    break

        for i, b in enumerate(blocks):
            b.index = i

            parents = [b]
            while (p := parents[-1].parent) is not None:
                parents.append(blocks[p])
            parents = parents[1:]

            b.is_local = not all(p.is_class for p in parents)
            b.is_method = not b.is_class and bool(parents) and parents[0].is_class

        def add_full_names(children: Sequence[Block], prefix: str = "") -> None:
            dupes: dict[str, list[Block]] = {}
            for b in children:
                dupes.setdefault(b.name, []).append(b)

            for dl in dupes.values():
                for i, b in enumerate(dl):
                    suffix = f"[{i + 1}]" if len(dl) > 1 else ""
                    b.full_name = prefix + b.name + suffix

            for b in children:
                if kids := [blocks[i] for i in b.children]:
                    add_full_names(kids, b.full_name + ".")

        add_full_names([b for b in blocks if b.parent is None])
        return blocks

    def block(self, begin: int) -> Block | None:
        t = self.tokens[begin]
        if not (t.type == token.NAME and t.string in ("class", "def")):
            return None

        category = Block.Category[t.string.upper()]
        try:
            ni = self.next_token(begin + 1, token.NAME, "Definition but no name")
            name = self.tokens[ni].string
            indent = self.next_token(ni + 1, token.INDENT, "Definition but no indent")
            dedent = self.indent_to_dedent[indent]
            docstring = self.docstring(indent)
        except _linter.ParseError:
            name = "(ParseError)"
            indent = -1
            dedent = -1
            docstring = ""

        return Block(
            begin=begin,
            category=category,
            dedent=dedent,
            docstring=docstring,
            indent=indent,
            name=name,
            tokens=self.tokens,
        )


class DocstringLinter(_linter.FileLinter[DocstringFile]):
    linter_name = "docstring_linter"
    description = DESCRIPTION
    is_fixer = False

    path_to_blocks: dict[str, list[dict[str, Any]]]
    path_to_errors: dict[str, list[dict[str, Any]]]

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        super().__init__(argv)
        add_arguments(self.parser.add_argument)
        self.path_to_blocks = {}
        self.path_to_errors = {}

    def lint_all(self) -> bool:
        success = super().lint_all()
        self._report()
        self._write_grandfather()
        return success

    def _lint(self, df: DocstringFile) -> Iterator[_linter.LintResult]:
        if (p := str(df.path)) in self.path_to_blocks:
            print("Repeated file", p, file=sys.stderr)
            return

        blocks = df.blocks
        bad = {b for b in blocks if self._is_bad_block(b, df)}
        bad = self._dont_require_constructor_and_class_docs(blocks, bad)
        gf = self._grandfathered(df.path, bad)

        yield from (self._block_result(b, df) for b in sorted(bad - gf))

        def as_data(b: Block) -> dict[str, Any]:
            status = "grandfather" if b in gf else "bad" if b in bad else "good"
            return {"status": status, **b.as_data()}

        self.path_to_blocks[p] = [as_data(b) for b in blocks]

    def _error(self, df: DocstringFile, result: _linter.LintResult) -> None:
        self.path_to_errors[str(df.path)] = [{str(result.line): result.name}]

    @cached_property
    def _grandfather(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self.args.grandfather) as fp:
                return json.load(fp)  # type: ignore[no-any-return]
        except FileNotFoundError:
            return {}
        except Exception as e:
            print("ERROR:", e, "in", GRANDFATHER_LIST, file=sys.stderr)
            raise

    @cached_property
    def _max_lines(self) -> dict[str, int]:
        return {"class": self.args.max_class, "def": self.args.max_def}

    def _grandfathered(self, path: Path | None, bad: set[Block]) -> set[Block]:
        if path is None or self.args.no_grandfather or self.args.write_grandfather:
            return set()

        grand: dict[str, int] = self._grandfather.get(str(path), {})
        tolerance_ratio = 1 + self.args.grandfather_tolerance / 100.0

        def grandfathered(b: Block) -> bool:
            lines = int(grand.get(b.display_name, 0) * tolerance_ratio)
            return b.line_count <= lines

        return {b for b in bad if grandfathered(b)}

    def _block_result(self, b: Block, df: DocstringFile) -> _linter.LintResult:
        def_name = "function" if b.category == "def" else "class"
        msg = f"docstring found for {def_name} '{b.name}' ({b.line_count} lines)"
        if len(b.docstring):
            s = "" if len(b.docstring) == 1 else "s"
            needed = f"needed {self.args.min_docstring}"
            msg = f"{msg} was too short ({len(b.docstring)} character{s}, {needed})"
        else:
            msg = f"No {msg}"
            if b.is_method:
                msg = f"{msg}. {METHOD_OVERRIDE_HINT}"
        return _linter.LintResult(msg, *df.tokens[b.begin].start)

    def _display(
        self, df: DocstringFile, results: list[_linter.LintResult]
    ) -> Iterator[str]:
        if not self.args.report:
            yield from super()._display(df, results)

    def _dont_require_constructor_and_class_docs(
        self, blocks: Sequence[Block], bad: set[Block]
    ) -> set[Block]:
        if self.args.lint_init:
            return bad

        good = {b for b in blocks if len(b.docstring) >= self.args.min_docstring}

        def has_class_init_doc(b: Block) -> bool:
            if b.is_class:
                # Is it a class whose constructor is documented?
                children = (blocks[i] for i in b.children)
                return any(b.is_init and b in good for b in children)

            # Is it a constructor whose class is documented?
            return b.is_init and b.parent is not None and blocks[b.parent] in good

        return {b for b in bad if not has_class_init_doc(b)}

    def _is_bad_block(self, b: Block, df: DocstringFile) -> bool:
        max_lines = self._max_lines[b.category]
        return (
            not df.omitted(df.tokens, b.begin, b.dedent)
            and b.line_count > max_lines
            and len(b.docstring) < self.args.min_docstring
            and (self.args.lint_local or not b.is_local)
            and (self.args.lint_protected or not b.name.startswith("_"))
        )

    def _report(self) -> None:
        if not self.args.lintrunner and self.path_to_blocks and self.args.report:
            report = {
                k: s for k, v in self.path_to_blocks.items() if (s := file_summary(v))
            } | self.path_to_errors
            print(json.dumps(report, sort_keys=True, indent=2))

    def _write_grandfather(self) -> None:
        if self.args.write_grandfather:
            results: dict[str, dict[str, int]] = {}

            for path, blocks in self.path_to_blocks.items():
                for block in blocks:
                    if block["status"] == "bad":
                        d = results.setdefault(path, {})
                        d[block["display_name"]] = block["line_count"]

            with open(self.args.grandfather, "w") as fp:
                json.dump(results, fp, sort_keys=True, indent=2)


def make_recursive(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rec(i: int) -> dict[str, Any]:
        d = dict(blocks[i])
        d["children"] = [rec(c) for c in d["children"]]
        return d

    return [rec(i) for i, b in enumerate(blocks) if b["parent"] is None]


def make_terse(
    blocks: Sequence[dict[str, Any]],
    index_by_line: bool = True,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    max_line = max(b["start_line"] for b in blocks) if blocks else 0
    line_field_width = len(str(max_line))

    for b in blocks:
        root = f"{b['category']} {b['full_name']}"
        for i in itertools.count():
            name = root + bool(i) * f"[{i + 1}]"
            if name not in result:
                break

        d = {
            "docstring_len": len(b["docstring"]),
            "lines": b["line_count"],
            "status": b.get("status", "good"),
        }

        start_line = b["start_line"]
        if index_by_line:
            d["name"] = name
            result[f"{start_line:>{line_field_width}}"] = d
        else:
            d["line"] = start_line
            result[name] = d

        if kids := b["children"]:
            if not all(isinstance(k, int) for k in kids):
                assert all(isinstance(k, dict) for k in kids)
                d["children"] = make_terse(kids)

    return result


def file_summary(
    blocks: Sequence[dict[str, Any]], report_all: bool = False
) -> dict[str, str]:
    def to_line(v: dict[str, Any]) -> str | None:
        if (status := v["status"]) == "good":
            if not report_all:
                return None
            fail = ""
        elif status == "grandfather":
            fail = ": (grandfathered)"
        else:
            assert status == "bad"
            fail = ": FAIL"
        name = v["name"]
        lines = v["lines"]
        docs = v["docstring_len"]
        parens = "()" if name.startswith("def ") else ""
        return f"{name}{parens}: {lines=}, {docs=}{fail}"

    t = make_terse(blocks)
    r = {k: line for k, v in t.items() if (line := to_line(v))}
    while r and all(k.startswith(" ") for k in r):
        r = {k[1:]: v for k, v in r.items()}
    return r


def add_arguments(add: Callable[..., Any]) -> None:
    h = "Set the grandfather list"
    add("--grandfather", "-g", default=str(GRANDFATHER_LIST), type=str, help=h)

    h = "Tolerance for grandfather sizes, in percent"
    add("--grandfather-tolerance", "-t", default=TOLERANCE_PERCENT, type=float, help=h)

    h = "Lint __init__ and class separately"
    add("--lint-init", "-i", action="store_true", help=h)

    h = "Lint definitions inside other functions"
    add("--lint-local", "-o", action="store_true", help=h)

    h = "Lint functions, methods and classes that start with _"
    add("--lint-protected", "-p", action="store_true", help=h)

    h = "Maximum number of lines for an undocumented class"
    add("--max-class", "-c", default=MAX_LINES["class"], type=int, help=h)

    h = "Maximum number of lines for an undocumented function"
    add("--max-def", "-d", default=MAX_LINES["def"], type=int, help=h)

    h = "Minimum number of characters for a docstring"
    add("--min-docstring", "-s", default=MIN_DOCSTRING, type=int, help=h)

    h = "Disable the grandfather list"
    add("--no-grandfather", "-n", action="store_true", help=h)

    h = "Print a report on all classes and defs"
    add("--report", "-r", action="store_true", help=h)

    h = "Rewrite the grandfather list"
    add("--write-grandfather", "-w", action="store_true", help=h)


if __name__ == "__main__":
    DocstringLinter.run()
