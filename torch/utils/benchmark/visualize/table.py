import abc
import dataclasses
import enum
import itertools as it
import re
from typing import Any, Dict, Callable, List, Optional, Pattern, Sequence, Tuple, TypeVar, Union

from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface

__all__ = ["Cell", "Layout", "Table"]


T = TypeVar("T")
REDUCTION_FUNCTION = Callable[[Tuple["Cell", ...]], T]
REDUCTION_OUTPUT = Tuple[T, int]
RESULT = Union[common.Measurement, timer_interface.CallgrindStats]
RESULTS = Sequence[RESULT]


# NB: We need to strip ANSI codes when computing width in order for
#     columns to align properly when printed in terminal.
# https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
_ANSI_CODE: Pattern = re.compile(r"\033\[[0-9]+(;[0-9]+)?m")


class Layout:
    """Container for various style control enums."""

    class HorizontalAlign(enum.Enum):
        LEFT = 0
        RIGHT = 1
        CENTER = 2

    class VerticalAlign(enum.Enum):
        TOP = 0
        BOTTOM = 1

    ALIGNMENT_PAIR = Tuple[HorizontalAlign, VerticalAlign]

    class ShouldRender(enum.Enum):
        YES = 0
        MAYBE = 1


class Cell(abc.ABC):
    """Core data container for `Table`. (And `Compare`)

    One of the primary duties of `Compare` is to data layout. Columns need to be
    aligned, empty cells need to be filled, row and column headers need to be
    set, etc. This work is largely orthogonal to determining the contents of
    the cells, except insofar as cell values might interact. (e.g. choosing a
    common unit or aligning decimal places for readability.)

    `Cell` attempts to separate these concerns by providing an interface where
    an author can define arbitrary data processing cells while deferring the
    more tedious and error-prone layout work to the framework.

    For simple use cases, simply define a `render()` method which returns a
    string representation of the cell (and optionally, override `alignment`
    if a different alignment is desired), and the table will collect and render
    the cells.

    More complicated use cases will involve interaction between cells. For
    instance, one needs to look at all entries of a row or column to color code
    appropriately. For these cases, the `row_reduce` and `col_reduce` methods
    can be used to implement cross cell data dependencies.
    """

    # TODO: Add `final` annotations once the minimum version is 3.8.
    _row: Optional["Row"] = None

    def __hash__(self) -> int:
        return hash(id(self))

    @abc.abstractmethod
    def render(self) -> str:
        ...

    @property
    def alignment(self) -> Layout.ALIGNMENT_PAIR:
        return Layout.HorizontalAlign.RIGHT, Layout.VerticalAlign.TOP

    @property
    def should_render(self) -> Layout.ShouldRender:
        return Layout.ShouldRender.YES

    def set_context(self, row: "Row") -> None:
        # We don't want to rely on __init__, as row is not known at construction.
        self._row = row

    def row_reduce(self, f: REDUCTION_FUNCTION) -> REDUCTION_OUTPUT:
        assert self._row is not None
        return self._row._row_reduce(self, f)

    def col_reduce(self, f: REDUCTION_FUNCTION) -> REDUCTION_OUTPUT:
        assert self._row is not None
        return self._row._table._col_reduce(self, f)

    def all_reduce(self, f: REDUCTION_FUNCTION) -> REDUCTION_OUTPUT:
        assert self._row is not None
        return self._row._table._all_reduce(self, f)


class FullRowCell(Cell):
    """This type of cell expects to occupy an entire row of a Table.

    Consequently, it also takes a `width` argument to `render` to that it can
    decide how to fill the row.

    All `FullRowCell` instances are considered to occupy a separate column.
    A subclass author may `col_reduce` over them, but that reduction will not
    include "normal" cells.
    """

    def render(self) -> str:
        raise NotImplementedError("`FullRowCell`s use the `render_row` method")

    @abc.abstractmethod
    def render_row(self, width: int) -> str:
        ...


class _ReductionGroup:
    """Maintain state for `row_reduce` and `col_reduce`."""
    _elements: Tuple[Cell, ...]
    _element_id: Dict[Cell, int]
    _cache: Dict[int, Any]

    def __init__(self, elements: Tuple[Cell, ...]) -> None:
        self._elements = elements
        self._element_id = {e: i for i, e in enumerate(elements)}
        self._cache = {}

    def __call__(self, cell: Cell, f: REDUCTION_FUNCTION) -> REDUCTION_OUTPUT:
        if id(f) not in self._cache:
            self._cache[id(f)] = f(self._elements)

        return self._cache[id(f)], self._element_id[cell]


@dataclasses.dataclass()
class _CellContents:
    """Convenience container for cell layout passes.

    Once a `Cell` renders its contents (which can be arbitrarily sized and
    arbitrarily many lines) the Table needs to modify that string so everything
    aligns properly. To facilitate this we store a Cell's rendered contents
    here so we can easily update them during the layout process.
    """

    lines: List[str]
    horizontal_alignment: Layout.HorizontalAlign
    vertical_alignment: Layout.VerticalAlign

    def pad_horizontal(self, n: int) -> None:
        raw_lines = [_ANSI_CODE.sub("", line) for line in self.lines]
        assert all(len(line) <= n for line in raw_lines)

        lines: List[str] = []
        for line, raw_line in zip(self.lines, raw_lines):
            pad_n = len(line) - len(raw_line) + n
            if self.horizontal_alignment == Layout.HorizontalAlign.LEFT:
                lines.append(line.ljust(pad_n))
            elif self.horizontal_alignment == Layout.HorizontalAlign.RIGHT:
                lines.append(line.rjust(pad_n))
            else:
                assert self.horizontal_alignment == Layout.HorizontalAlign.CENTER
                lines.append(line.center(pad_n))
        self.lines = lines

    def pad_vertical(self, n: int) -> None:
        assert len(self.lines) <= n, f"{n}, {len(self.lines)}"
        upper_padding, lower_padding = {
            Layout.VerticalAlign.TOP: (0, n - len(self.lines)),
            Layout.VerticalAlign.BOTTOM: (n - len(self.lines), 0),
        }[self.vertical_alignment]
        self.lines = upper_padding * [""] + self.lines + lower_padding * [""]

    @property
    def width(self) -> int:
        return max(len(_ANSI_CODE.sub("", line)) for line in self.lines)


class Row:
    """A collection of Cells.

    The Row is responsible for:
      1) Vertically aligning rendered cell contents.
      2) Exposing the parent Table to Cells which need to `col_reduce`.
      3) Implementing `row_reduce` for cells. (Which is much simpler and easier
         than `col_reduce` as rows are explicitly represented.)

    Table is responsible for much more of the layout burden, and Row is
    principally a glue class.
    """

    _cells: Tuple[Cell, ...]
    _reduction_group: _ReductionGroup
    _table: "Table"

    def __init__(self, cells: Tuple[Cell, ...], table: "Table") -> None:
        self._cells = cells
        self._table = table
        self._reduction_group = _ReductionGroup(cells)
        for cell in cells:
            cell.set_context(row=self)

    def __len__(self) -> int:
        return len(self._cells)

    def render(self) -> List[_CellContents]:
        cell_lines: List[_CellContents] = [
            _CellContents(
                cell.render().splitlines(keepends=False),
                *cell.alignment
            )
            for cell in self._cells
        ]

        row_height = max(len(cl.lines) for cl in cell_lines)
        for cl in cell_lines:
            cl.pad_vertical(row_height)

        return cell_lines

    @property
    def should_render(self) -> bool:
        return any(c.should_render == Layout.ShouldRender.YES for c in self._cells)

    def _row_reduce(self, cell: Cell, f: REDUCTION_FUNCTION) -> REDUCTION_OUTPUT:
        return self._reduction_group(cell, f)


class Table:
    """This class is responsible for final layout.

    When laying out rendered cell contents, the final cell representation will
    depend on both all other elements in the row or column. This could affect
    the cell (e.g. to colorize values) or it could affect whether the cell
    should be rendered at all.

    Table does not support `rowspan` and `colspan`, however a special
    `FullRowCell` case is supported to implement section partitions.
    """

    _rows: List[Row]
    _row_scaffold: Dict[int, FullRowCell]
    _should_render_row: Tuple[bool, ...]
    _should_render_col: Tuple[bool, ...]
    _reduction_groups: Dict[Cell, _ReductionGroup]
    _global_reduction: _ReductionGroup

    def __init__(
        self,
        rows: Sequence[Union[FullRowCell, Sequence[Cell]]],
    ) -> None:
        self._rows = []
        self._row_scaffold = {}

        for i, r in enumerate(rows):
            if isinstance(r, FullRowCell):
                self._row_scaffold[i] = r
                _ = Row((r,), table=self)  # Link for `col_reduce`

            else:
                assert isinstance(r, (tuple, list))
                assert all(isinstance(c, Cell) for c in r)
                self._rows.append(Row(tuple(r), table=self))

        self._should_render_row = tuple(row.should_render for row in self._rows)

        # zip will silently drop elements from longer inputs, so we explicitly
        # check the sizes.
        n_cols = common.get_unique([len(row) for row in self._rows])
        assert n_cols > 0, f"{n_cols}"

        columns = tuple(zip(*[row._cells for row in self._rows]))
        self._should_render_col = tuple(
            any(c.should_render == Layout.ShouldRender.YES for c in col)
            for col in columns
        )

        # Handle bookkeeping for `FullRowCell`s (table scaffolding) and `col_reduce`
        scaffold_reduction = _ReductionGroup(tuple(self._row_scaffold.values()))
        self._reduction_groups = {c: scaffold_reduction for c in self._row_scaffold.values()}

        self._global_reduction = _ReductionGroup(
            tuple(it.chain(*[[r] if isinstance(r, Cell) else r for r in rows])))

        for column in columns:
            reduction_group = _ReductionGroup(column)
            self._reduction_groups.update({c: reduction_group for c in column})

    def _col_reduce(self, cell: Cell, f: REDUCTION_FUNCTION) -> REDUCTION_OUTPUT:
        return self._reduction_groups[cell](cell, f)

    def _all_reduce(self, cell: Cell, f: REDUCTION_FUNCTION) -> REDUCTION_OUTPUT:
        return self._global_reduction(cell, f)

    def render(self) -> str:
        raw_contents = [
            [
                cell_contents
                for i_col, cell_contents in enumerate(r.render())
                if self._should_render_col[i_col]
            ]
            for i_row, r in enumerate(self._rows)
            if self._should_render_row[i_row]
        ]

        # `raw_contents` is a rowwise storage, we transpose to columnwise for
        # horizontal padding.
        for column in zip(*raw_contents):
            width = max(c.width for c in column)
            for c in column:
                c.pad_horizontal(width)

        # At this point the cell contents have been vertically aligned by Row,
        # and we just horizontally aligned them, so they're ready to be
        # converted into raw strings.
        data_lines: List[str] = []
        for row in raw_contents:
            for lines in zip(*[cell_contents.lines for cell_contents in row]):
                data_lines.append(" | ".join(lines))
        data_lines.reverse()

        table_width = common.get_unique([len(_ANSI_CODE.sub("", l)) for l in data_lines])
        table_lines: List[str] = []
        for i in range(len(data_lines) + len(self._row_scaffold)):
            row_cell = self._row_scaffold.get(i)
            if row_cell is None:
                # Normal Row
                table_lines.append(data_lines.pop())

            elif row_cell.should_render == Layout.ShouldRender.YES:
                # FullRowCell
                table_lines.append(row_cell.render_row(width=table_width))

        return "\n".join(table_lines)
