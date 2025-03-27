from __future__ import annotations

import dataclasses as dc
from enum import Enum
from functools import total_ordering
from typing import Any, TYPE_CHECKING
from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Sequence
    from tokenize import TokenInfo


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
    parent: int | None = None

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

    DATA_FIELDS = (
        "category",
        "children",
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


def make_blocks(tokens: Sequence[TokenInfo]) -> tuple[list[Block], dict[str, str]]:
    blocks: list[Block] = []
    errors: dict[str, str] = {}

    for i in range(len(tokens)):
        try:
            if (b := _make_block(tokens, i)) is not None:
                blocks.append(b)
        except ParseError as e:
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


def _make_block(tokens: Sequence[TokenInfo], begin: int) -> Block | None:
    t = tokens[begin]
    if not (t.type == token.NAME and t.string in ("class", "def")):
        return None

    category = Block.Category[t.string.upper()]
    try:
        ni = self.next_token(begin + 1, token.NAME, "Definition but no name")
        name = self.tokens[ni].string
        indent = self.next_token(ni + 1, token.INDENT, "Definition but no indent")
        dedent = self.indent_to_dedent[indent]
        docstring = self.docstring(indent)
    except ParseError:
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
