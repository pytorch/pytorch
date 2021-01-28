"""Handle rendering benchmark results.

TODO:
    Most of this machinery should be rolled into `Compare` to make it more
    general and robust, but for now it is a standalone implementation.
"""
import abc
import collections
import dataclasses
import enum
import itertools as it
import re
import textwrap
from typing import (
    cast, Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Set,
    Tuple, Union, TYPE_CHECKING
)

from core.api import AutogradMode, AutoLabels, RuntimeMode
from core.types import Label

if TYPE_CHECKING:
    # See core.api for an explanation why this is necessary.
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Language, Measurement


ValueType = Tuple[CallgrindStats, Measurement]
ResultType = Tuple[Tuple[Label, int, AutoLabels, ValueType], ...]


class Alignment(enum.Enum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2
    ANY = 3


class ShouldRender(enum.Enum):
    YES = 0
    MAYBE = 1


FAINT = "\033[2m"
BOLD = "\033[1m"
LIGHT_RED = "\033[1;31m"
LIGHT_GREEN = "\033[1;32m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
END_CHAR = "\033[0m"
PYTHON_DIM_THRESHOLD = 0.001  # 0.1%
COLOR_THRESHOLDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("  0.0%", "{}{}", "", ""),
    ("  0.5%", f"{FAINT}{{}}{{}}{END_CHAR * 2}", LIGHT_GREEN, LIGHT_RED),
    ("  2.5%", f"{{}}{{}}{END_CHAR}", LIGHT_GREEN, LIGHT_RED),
    (" 10.0%", f"{BOLD}{{}}{{}}{END_CHAR * 2}", LIGHT_GREEN, LIGHT_RED),
    (" 25.0%", f"{FAINT}{{}}{{}}{END_CHAR * 2}", GREEN, RED),
    (" 50.0%", f"{{}}{{}}{END_CHAR}", GREEN, RED),
    ("100.0%", f"{BOLD}{{}}{{}}{END_CHAR * 2}", GREEN, RED),
)

# This is assumed by later parsing.
assert all(t.endswith("%") for t, _, _, _ in COLOR_THRESHOLDS)


def strip_ansi(s: str) -> str:
    # https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
    return re.sub(r"\033\[[0-9]+(;[0-9]+)?m", "", s)


def pad(s: str, width: int, alignment: Alignment) -> str:
    width += (len(s) - len(strip_ansi(s)))
    if alignment == Alignment.LEFT:
        return s.ljust(width)
    elif alignment == Alignment.RIGHT:
        return s.rjust(width)
    else:
        assert alignment == Alignment.CENTER
        return s.center(width)


class Cell(abc.ABC):
    # TODO: annotate with `Final` and `final`.
    _row: Optional["Row"] = None

    def set_context(self, row: "Row") -> None:
        # We don't want to rely on __init__, as Row is private.
        # (And not necessarily known at construction.)
        self._row = row

    def col_reduce(self, f: Callable[[Tuple["Cell", ...]], Any]) -> Tuple[Any, int]:
        assert self._row is not None
        return self._row._table.col_reduce(self, f)

    @abc.abstractmethod
    def render(self) -> Union[str, Tuple[str, int]]:
        ...

    @abc.abstractmethod
    def alignment(self) -> Alignment:
        ...

    @property
    def should_render(self) -> ShouldRender:
        return ShouldRender.YES


@dataclasses.dataclass(frozen=True)
class Row:
    cells: Tuple[Cell, ...]
    _table: "Table"

    def __len__(self) -> int:
        return len(self.cells)

    def render(self) -> Tuple[Tuple[str, ...], ...]:
        col_renders: List[Tuple[Tuple[str, ...], int]] = []
        padding = [0, 0]
        for cell in self.cells:
            r: Union[str, Tuple[str, int]] = cell.render()
            r_str, ref_index = (r, 0) if isinstance(r, str) else r
            lines = tuple(r_str.splitlines(keepends=False))
            ref_index %= len(lines) or 1  # Handle negative indexing
            col_renders.append((lines, ref_index))

            padding[0] = max(padding[0], ref_index)
            padding[1] = max(padding[1], len(lines) - ref_index)

        output: List[Tuple[str, ...]] = []
        for lines, ref_index in col_renders:
            top_pad = padding[0] - ref_index
            bottom_pad = padding[1] - (len(lines) - ref_index)
            output.append(("",) * top_pad + lines + ("",) * bottom_pad)
        return tuple(output)


class Table:
    _rows: Tuple[Row, ...]
    _reduction_cache: Dict[Tuple[int, int], Any]
    _row_index_map: Dict[int, int]
    _col_index_map: Dict[int, int]

    def __init__(self, rows: Iterable[Tuple[Cell, ...]]):
        self._reduction_cache = {}
        self._rows = tuple(Row(r, self) for r in rows)
        self._row_index_map = {}
        self._col_index_map = {}
        for i_row, r in enumerate(self._rows):
            for i_col, cell in enumerate(r.cells):
                self._row_index_map[id(cell)] = i_row
                self._col_index_map[id(cell)] = i_col
                cell.set_context(r)

        n_cols = set(len(r) for r in self._rows)
        assert len(n_cols) == 1, f"{n_cols}"
        self._n_cols: int = n_cols.pop()
        assert self._n_cols > 0

        self._render_col: Tuple[bool, ...] = tuple(
            any(c.should_render == ShouldRender.YES for c in col)
            for col in zip(*[r.cells for r in self._rows])
        )

    def col_reduce(self, caller: Cell, f: Callable[[Tuple["Cell", ...]], Any]) -> Tuple[Any, int]:
        caller_id = id(caller)
        column_index = self._col_index_map[caller_id]
        key = (column_index, id(f))
        if key not in self._reduction_cache:
            self._reduction_cache[key] = f(tuple(r.cells[column_index] for r in self._rows))

        return self._reduction_cache[key], self._row_index_map[caller_id]

    @property
    def col_separator(self) -> str:
        return "  ┊  "

    def render(self) -> Tuple[str, ...]:
        segments: Tuple[Tuple[Tuple[str, ...], ...], ...] = tuple(
            r.render() for r in self._rows)

        alignments: List[Alignment] = [Alignment.ANY for _ in range(self._n_cols)]
        for r in self._rows:
            for i, cell in enumerate(r.cells):
                if alignments[i] == Alignment.ANY:
                    alignments[i] = cell.alignment()
                    continue

                assert cell.alignment() in (alignments[i], Alignment.ANY)

        alignments = [Alignment.CENTER if a == Alignment.ANY else a for a in alignments]
        col_widths: Tuple[int, ...] = tuple(
            max(len(strip_ansi(s)) for s in it.chain(*col_segments))
            for col_segments in zip(*segments))

        rendered_rows: List[str] = []
        for row_segments in segments:
            padded_row: List[Tuple[str, ...]] = []
            for j in zip(row_segments, col_widths, alignments, self._render_col):
                col, width, alignment, render_col = j
                if render_col:
                    padded_row.append(tuple(pad(s, width, alignment) for s in col))

            rendered_rows.append("\n".join([
                self.col_separator.join(ri) for ri in zip(*padded_row)
            ]))

        return tuple(rendered_rows)

    def __str__(self) -> str:
        return "\n".join(self.render())


class Null_Cell(Cell):
    def render(self) -> str:
        return ""

    def alignment(self) -> Alignment:
        return Alignment.ANY

    @property
    def should_render(self) -> ShouldRender:
        return ShouldRender.MAYBE


class ColHeader_Cell(Cell):
    def __init__(self, header: str) -> None:
        self._header = header

    def render(self) -> str:
        return self._header

    def alignment(self) -> Alignment:
        return Alignment.ANY

    @property
    def should_render(self) -> ShouldRender:
        return ShouldRender.MAYBE


class Label_Cell(Cell):
    def __init__(self, label: Label, autograd: AutogradMode) -> None:
        self._label: Label = label
        self._autograd: AutogradMode = autograd

    def render(self) -> Tuple[str, int]:
        masks, i = self.col_reduce(Label_Cell.gather_masks)
        mask = masks[i]
        assert len(mask) == len(self.label)
        result = "\n".join([
            "  " * i + li
            for i, (mi, li) in enumerate(zip(mask, self.label))
            if mi
        ]) or ""

        # Extra space for top level labels.
        if mask[0] and any(len(m) > 1 for m in masks):
            result = f"\n{result}"

        return result, -1

    def alignment(self) -> Alignment:
        return Alignment.LEFT

    @property
    def label(self) -> Label:
        render_autograd, i = self.col_reduce(self.should_render_autograd)
        autograd_repr = {
            AutogradMode.FORWARD: "Mode: Forward",
            AutogradMode.FORWARD_BACKWARD: "Mode: Forward + Backward",
            AutogradMode.EXPLICIT: " ",
        }
        return self._label + (
            (autograd_repr[self._autograd],) if render_autograd[i] else ())

    @staticmethod
    def gather_masks(col: Tuple[Cell, ...]) -> Tuple[Tuple[bool, ...], ...]:
        masks: List[Tuple[bool, ...]] = []
        prior_l: Label = ()

        for c in col:
            if not isinstance(c, Label_Cell):
                masks.append(())
                continue

            l: Label = c.label
            i: int = 0
            for i, (l_i, pl_i) in enumerate(zip(l, prior_l)):
                if l_i != pl_i:
                    break
                i += 1
            prior_l = l
            masks.append((False,) * i + (True,) * (len(l) - i))

        return tuple(masks)

    @staticmethod
    def should_render_autograd(col: Tuple[Cell, ...]) -> Tuple[bool, ...]:
        autograd_set: DefaultDict[Label, Set[AutogradMode]] = collections.defaultdict(set)
        for c in col:
            if isinstance(c, Label_Cell):
                autograd_set[c._label].add(c._autograd)

        return tuple(
            len(autograd_set[c._label]) > 1 if isinstance(c, Label_Cell) else False
            for c in col
        )

class NumThreads_Cell(Cell):
    def __init__(self, num_threads: int) -> None:
        self._num_threads: int = num_threads

    def render(self) -> str:
        return f"{self._num_threads}  "

    def alignment(self) -> Alignment:
        return Alignment.RIGHT


class AB_Cell(Cell):
    def __init__(
        self,
        a: ValueType,
        b: ValueType,
        language: Language,
        display_time: bool = False,
        colorize: bool = False,
    ) -> None:
        self._display_time: bool = display_time
        significant_figures = min(
            a[1].significant_figures,
            b[1].significant_figures,
        )
        self._significant_figures = significant_figures

        a_t, b_t = a[1].median, b[1].median
        delta_t = abs(b_t - a_t) / b_t
        self._zero_within_noise: bool = (delta_t < 1.0 / (significant_figures or 1))
        self._robust_time = (
            display_time and significant_figures and not self._zero_within_noise
        )
        self._a_times: float = a_t * 1e6
        self._b_times: float = b_t * 1e6

        self._a_counts = int(a[0].counts(denoise=True) / a[0].number_per_run)
        self._b_counts = int(b[0].counts(denoise=True) / b[0].number_per_run)

        self._language = language
        self._colorize = colorize

    def render(self) -> str:
        if self._a_counts == self._b_counts:
            return "..."

        segment_lengths, _ = self.col_reduce(AB_Cell.segment_lengths)
        i_s0, i_s1, i_s2, t_s0, t_s1, t_s2 = [
            pad(s, l, Alignment.RIGHT)
            for s, l in zip(self.segments, segment_lengths * 2)
        ]

        output: str = f"{i_s0} -> {i_s1}  {i_s2}"
        instr_abs_delta = abs(self._a_counts - self._b_counts) / ((self._a_counts + self._b_counts) / 2)
        if self._language == Language.PYTHON and instr_abs_delta < PYTHON_DIM_THRESHOLD:
            output = f"{FAINT}{output}{END_CHAR}"
        if self._display_time:
            time_str = f"{t_s0} -> {t_s1}  {t_s2}"
            if self._zero_within_noise:
                time_str = f"{FAINT}{time_str}{END_CHAR}"

            output = f"{output}\n{time_str}\n "

        return output

    def make_segments(
        self,
        a: Union[int, float],
        b: Union[int, float],
        template: str = "{}{}"
    ) -> Tuple[str, str, str]:
        sign_str = "+" if b >= a else "-"
        return (
            template.format(sign_str, abs(b - a)),
            template.format("", b),
            self.maybe_colorize_delta(sign_str, abs(b - a) / b)
        )

    def maybe_colorize_delta(self, sign_str: str, value: float) -> str:
        value *= 100  # Convert to percent
        template = "({}{:.1f}%)"
        if not self._colorize:
            return template.format(sign_str, value)

        color_template, good_color, bad_color = [
            i[1:]
            for i in COLOR_THRESHOLDS
            if abs(value) >= (float(i[0].strip("%")))
        ][-1]

        return color_template.format(
            bad_color if sign_str == "+" else good_color,
            template.format(sign_str, value)
        )

    @property
    def segments(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.make_segments(self._a_counts, self._b_counts, "{}{}") +
            self.make_segments(self._a_times, self._b_times, "{}{:.1f}")
        )

        i_segments = self.make_segments(self._a_counts, self._b_counts, "{}{}")
        t_segments = (
            self.make_segments(self._a_times, self._b_times, "{}{:.1f}")
            if self._robust_time else ("", "", "")
        )
        return i_segments + t_segments

    @staticmethod
    def segment_lengths(col: Tuple[Cell, ...]) -> Tuple[int, int, int]:
        lengths = [0, 0, 0, 0, 0, 0]
        for c in col:
            if isinstance(c, AB_Cell):
                lengths = [max(li, len(strip_ansi(si))) for li, si in zip(lengths, c.segments)]

        output = tuple(max(i, j) for i, j in zip(lengths[:3], lengths[3:]))
        assert len(output) == 3
        return cast(Tuple[int, int, int], output)

    def alignment(self) -> Alignment:
        return Alignment.RIGHT


def render_ab(
    a_results: ResultType,
    b_results: ResultType,
    display_time: bool = False,
    colorize: bool = False,
) -> None:
    packed_results: Dict[
        Tuple[Label, int, AutogradMode],
        Dict[Tuple[Language, RuntimeMode], Tuple[ValueType, ValueType]]
    ] = {}

    assert len(a_results) == len(b_results)
    for a_result, b_result in zip(a_results, b_results):
        assert a_result[:3] == b_result[:3]

        label, num_threads, auto_labels, a_value = a_result
        b_value = b_result[3]

        primary_key = (label, num_threads, auto_labels.autograd)
        secondary_key = (auto_labels.language, auto_labels.runtime)
        packed_results.setdefault(primary_key, {})
        assert secondary_key not in packed_results[primary_key]
        packed_results[primary_key][secondary_key] = (a_value, b_value)

    col_titles = {
        (Language.PYTHON, RuntimeMode.EAGER): "Python (Eager)",
        (Language.PYTHON, RuntimeMode.JIT): "Python (TorchScript)",
        (Language.PYTHON, RuntimeMode.EXPLICIT): "Python",
        (Language.CPP, RuntimeMode.EAGER): "C++ (Eager)",
        (Language.CPP, RuntimeMode.JIT): "C++ (TorchScript)",
        (Language.CPP, RuntimeMode.EXPLICIT): "C++",
    }

    column_keys = tuple((lang, rt) for lang in Language for rt in RuntimeMode)
    rows: List[Tuple[Cell, ...]] = [
        (Null_Cell(), ColHeader_Cell(header="num  \nthreads")) +
        tuple(ColHeader_Cell(header=col_titles[lang_rt]) for lang_rt in column_keys)
    ]

    for (label, num_threads, autograd), r in packed_results.items():
        rows.append((
            Label_Cell(label=label, autograd=autograd),
            NumThreads_Cell(num_threads=num_threads),
        ) + tuple(
            AB_Cell(*r[lang_rt], lang_rt[0], display_time, colorize) if lang_rt in r else Null_Cell()
            for lang_rt in column_keys
        ))

    table = Table(rows)
    print(str(table))
    print(textwrap.dedent("""
    Cell format:
        `\u0394I -> final I (% change)`
        For example, if A is 150 instructions and B is 144
        this would be represented: `-6 -> 144  (-4.1%)`
    """))
