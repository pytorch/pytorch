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
