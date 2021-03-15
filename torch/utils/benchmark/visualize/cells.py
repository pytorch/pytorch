import dataclasses
from typing import Callable, Dict, Optional, Sequence, Tuple, Type

import numpy as np

from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface
from torch.utils.benchmark.visualize.table import Cell, FullRowCell, Layout, REDUCTION_FUNCTION, REDUCTION_OUTPUT

__all__ = ["DataCellBase", "DataCellConfig", "DataCell"]


BEST = "\033[92m"
GOOD = "\033[34m"
BAD = "\033[91m"
VERY_BAD = "\033[31m"
BOLD = "\033[1m"
FAINT = "\033[2m"
TERMINATE = "\033[0m"


@dataclasses.dataclass(frozen=True)
class DataCellConfig:
    rowwise: bool
    colorize: bool
    trim_significant_figures: bool
    highlight_warnings: bool


class DataCellBase(Cell):
    def __init__(
        self,
        wall_times: Optional[Tuple[common.Measurement, ...]],
        instruction_counts: Optional[Tuple[timer_interface.CallgrindStats, ...]],
        config: DataCellConfig,
    ) -> None:
        ...

    @staticmethod
    def render_footer(cells: Sequence[Cell]) -> str:
        return ""


class NullCell(Cell):
    """Placeholder for cells in the Table which have no value."""
    def render(self) -> str:
        return ""


class TitleHeader(FullRowCell):
    def __init__(self, title: Optional[str]) -> None:
        self._title = title

    def render_row(self, width: int) -> str:
        return "[" + f" {self._title or ''} ".center(width - 2, "-") + "]"

    @property
    def should_render(self) -> Layout.ShouldRender:
        return Layout.ShouldRender.MAYBE if self._title is None else Layout.ShouldRender.YES


class NumThreadsHeader(FullRowCell):
    def __init__(self, n: int) -> None:
        self._n = n

    def render_row(self, width: int) -> str:
        return f"{self._n} threads: ".ljust(width, "-")

    @property
    def should_render(self) -> Layout.ShouldRender:
        should_render, i = self.col_reduce(self._determine_should_render)
        return Layout.ShouldRender.YES if should_render[i] else Layout.ShouldRender.MAYBE

    @staticmethod
    def _determine_should_render(cells: Sequence[Cell]) -> Tuple[bool, ...]:
        num_threads = tuple(
            cell._n if isinstance(cell, NumThreadsHeader) else None
            for cell in cells
        )

        if len(set(n for n in num_threads if n is not None)) < 2:
            return tuple(False for _ in cells)

        return tuple(
            isinstance(v, int) and v != v_prior
            for v, v_prior in zip(num_threads, (None,) + num_threads)
        )


class FallbackHeader(FullRowCell):
    def render_row(self, width: int) -> str:
        return "-" * width

    @property
    def should_render(self) -> Layout.ShouldRender:
        render_threads, _ = self.col_reduce(NumThreadsHeader._determine_should_render)
        return Layout.ShouldRender.MAYBE if any(render_threads) else Layout.ShouldRender.YES


class RowHeader(Cell):
    def __init__(self, env: Optional[str], row_key: str) -> None:
        self._env = env
        self._row_key = row_key

    def render(self) -> str:
        (max_pad, render_cell), i = self.col_reduce(self._get_env_padding)
        env_str = (f"({self._env})" if render_cell[i] else "").ljust(max_pad)
        return f"{env_str}{self._row_key}"

    @property
    def alignment(self) -> Layout.ALIGNMENT_PAIR:
        return Layout.HorizontalAlign.LEFT, Layout.VerticalAlign.TOP

    @staticmethod
    def _get_env_padding(cells: Sequence[Cell]) -> Tuple[int, Tuple[bool, ...]]:
        env_values = tuple(
            cell._env if isinstance(cell, RowHeader) else ""
            for cell in cells
        )

        max_pad = max(len(v or "") for v in env_values)
        if not max_pad:
            return 0, tuple(False for _ in env_values)

        return max_pad + 4, tuple(
            v != v_prior
            for v, v_prior in zip(env_values, ("",) + env_values)
        )


class ColHeader(Cell):
    def __init__(self, col_key: str) -> None:
        self._col_key = col_key

    def render(self) -> str:
        return self._col_key


class FooterCell(FullRowCell):
    def __init__(self, data_cell: Type[DataCellBase]):
        self._data_cell = data_cell

    def render_row(self, width: int) -> str:
        lines, _ = self.all_reduce(self._data_cell.render_footer)
        return "\n".join([l.ljust(width) for l in lines.splitlines(keepends=False)])

    @property
    def should_render(self) -> Layout.ShouldRender:
        lines, _ = self.all_reduce(self._data_cell.render_footer)
        return Layout.ShouldRender.YES if lines else Layout.ShouldRender.MAYBE


class DataCell(DataCellBase):
    def __init__(
        self,
        wall_times: Optional[Tuple[common.Measurement, ...]],
        instruction_counts: Optional[Tuple[timer_interface.CallgrindStats, ...]],
        config: DataCellConfig,
    ) -> None:
        self._config = config

        self._wall_times: Optional[common.Measurement] = None
        if wall_times:
            merged_times = common.Measurement.merge(wall_times)
            assert len(merged_times) == 1
            self._wall_times = merged_times[0]

        # Instruction counts are currently ignored.

    def render(self) -> str:
        if self._wall_times is None:
            return ""

        multiplier, decimal_digits, min_times, any_has_warnings = self.layout
        t = self._wall_times.median
        if self._config.trim_significant_figures:
            t = common.trim_sigfig(t, self._wall_times.significant_figures)
        else:
            decimal_digits = 1

        t_str = f"{t / multiplier:>.{decimal_digits}f}"

        if self._config.colorize:
            t_min = min_times[self._wall_times.num_threads]
            if t <= t_min * 1.01 or t <= t_min + 100e-9:
                t_str = f"{BEST}{BOLD}{t_str}{TERMINATE * 2}"
            elif t <= t_min * 1.1:
                t_str = f"{GOOD}{BOLD}{t_str}{TERMINATE * 2}"
            elif t >= t_min * 5:
                t_str = f"{VERY_BAD}{BOLD}{t_str}{TERMINATE * 2}"
            elif t >= t_min * 2:
                t_str = f"{FAINT}{BAD}{t_str}{TERMINATE * 2}"

        if self._config.highlight_warnings and any_has_warnings:
            warn_str = (
                f" (! {self._wall_times.iqr / self._wall_times.median * 100:.0f}%)"
                if self._wall_times.has_warnings else ""
            )
            t_str = f"{t_str}{warn_str:>9}"

        return f"{t_str} "

    @staticmethod
    def render_footer(cells: Sequence[Cell]) -> str:
        time_unit, _ = DataCell._select_unit(cells)
        return f"\nTimes are in {common.unit_to_english(time_unit)}s ({time_unit})."

    @property
    def reduce(self) -> Callable[[REDUCTION_FUNCTION], REDUCTION_OUTPUT]:
        return self.row_reduce if self._config.rowwise else self.col_reduce

    @staticmethod
    def _extract_times(cells: Sequence[Cell]) -> Tuple[common.Measurement, ...]:
        return tuple(
            cell._wall_times for cell in cells
            if isinstance(cell, DataCell) and cell._wall_times is not None
        )

    @staticmethod
    def _select_unit(cells: Sequence[Cell]) -> Tuple[str, float]:
        return common.select_unit(min([
            t.median for t in DataCell._extract_times(cells)
        ]))

    @property
    def layout(self) -> Tuple[float, int, Dict[int, float], bool]:
        (_, scale_factor), _ = self.all_reduce(self._select_unit)
        times, _ = self.reduce(self._extract_times)
        decimal_digits = min([
            t.significant_figures - int(np.ceil(np.log10(t.median / scale_factor)))
            for t in times
        ])

        min_times: Dict[int, float] = {}
        for t in times:
            min_times[t.num_threads] = min(t.median, min_times.get(t.num_threads, t.median))

        return scale_factor, max(decimal_digits, 0), min_times, any(t.has_warnings for t in times)
