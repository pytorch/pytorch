"""Display class to aggregate and print the results of many measurements."""
import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple

from torch.utils.benchmark.utils import common
from torch import tensor as _tensor

__all__ = ["Compare"]

BEST = "\033[92m"
GOOD = "\033[34m"
BAD = "\033[2m\033[91m"
VERY_BAD = "\033[31m"
BOLD = "\033[1m"
TERMINATE = "\033[0m"


class Colorize(enum.Enum):
    NONE = "none"
    COLUMNWISE = "columnwise"
    ROWWISE = "rowwise"


# Classes to separate internal bookkeeping from what is rendered.
class _Column(object):
    def __init__(
        self,
        grouped_results: List[Tuple[Optional[common.Measurement], ...]],
        time_scale: float,
        time_unit: str,
        trim_significant_figures: bool,
        highlight_warnings: bool,
    ):
        self._grouped_results = grouped_results
        self._flat_results = list(it.chain(*grouped_results))
        self._time_scale = time_scale
        self._time_unit = time_unit
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = (
            highlight_warnings
            and any(r.has_warnings for r in self._flat_results if r)
        )
        leading_digits = [
            int(_tensor(r.median / self._time_scale).log10().ceil()) if r else None
            for r in self._flat_results
        ]
        unit_digits = max(d for d in leading_digits if d is not None)
        decimal_digits = min(
            max(m.significant_figures - digits, 0)
            for digits, m in zip(leading_digits, self._flat_results)
            if (m is not None) and (digits is not None)
        ) if self._trim_significant_figures else 1
        length = unit_digits + decimal_digits + (1 if decimal_digits else 0)
        self._template = f"{{:>{length}.{decimal_digits}f}}{{:>{7 if self._highlight_warnings else 0}}}"

    def get_results_for(self, group):
        return self._grouped_results[group]

    def num_to_str(self, value: Optional[float], estimated_sigfigs: int, spread: Optional[float]):
        if value is None:
            return " " * len(self.num_to_str(1, estimated_sigfigs, None))

        if self._trim_significant_figures:
            value = common.trim_sigfig(value, estimated_sigfigs)

        return self._template.format(
            value,
            f" (! {spread * 100:.0f}%)" if self._highlight_warnings and spread is not None else "")


def optional_min(seq):
    l = list(seq)
    return None if len(l) == 0 else min(l)


class _Row(object):
    def __init__(self, results, row_group, render_env, env_str_len,
                 row_name_str_len, time_scale, colorize, num_threads=None):
        super(_Row, self).__init__()
        self._results = results
        self._row_group = row_group
        self._render_env = render_env
        self._env_str_len = env_str_len
        self._row_name_str_len = row_name_str_len
        self._time_scale = time_scale
        self._colorize = colorize
        self._columns: Tuple[_Column, ...] = ()
        self._num_threads = num_threads

    def register_columns(self, columns: Tuple[_Column, ...]):
        self._columns = columns

    def as_column_strings(self):
        concrete_results = [r for r in self._results if r is not None]
        env = f"({concrete_results[0].env})" if self._render_env else ""
        env = env.ljust(self._env_str_len + 4)
        output = ["  " + env + concrete_results[0].as_row_name]
        for m, col in zip(self._results, self._columns or ()):
            if m is None:
                output.append(col.num_to_str(None, 1, None))
            else:
                output.append(col.num_to_str(
                    m.median / self._time_scale,
                    m.significant_figures,
                    m.iqr / m.median if m.has_warnings else None
                ))
        return output

    @staticmethod
    def color_segment(segment, value, best_value):
        if value <= best_value * 1.01 or value <= best_value + 100e-9:
            return BEST + BOLD + segment + TERMINATE * 2
        if value <= best_value * 1.1:
            return GOOD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 5:
            return VERY_BAD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 2:
            return BAD + segment + TERMINATE * 2

        return segment

    def row_separator(self, overall_width):
        return (
            [f"{self._num_threads} threads: ".ljust(overall_width, "-")]
            if self._num_threads is not None else []
        )

    def finalize_column_strings(self, column_strings, col_widths):
        best_values = [-1 for _ in column_strings]
        if self._colorize == Colorize.ROWWISE:
            row_min = min(r.median for r in self._results if r is not None)
            best_values = [row_min for _ in column_strings]
        elif self._colorize == Colorize.COLUMNWISE:
            best_values = [
                optional_min(r.median for r in column.get_results_for(self._row_group) if r is not None)
                for column in (self._columns or ())
            ]

        row_contents = [column_strings[0].ljust(col_widths[0])]
        for col_str, width, result, best_value in zip(column_strings[1:], col_widths[1:], self._results, best_values):
            col_str = col_str.center(width)
            if self._colorize != Colorize.NONE and result is not None and best_value is not None:
                col_str = self.color_segment(col_str, result.median, best_value)
            row_contents.append(col_str)
        return row_contents


class Table(object):
    def __init__(
            self,
            results: List[common.Measurement],
            colorize: Colorize,
            trim_significant_figures: bool,
            highlight_warnings: bool
    ):
        assert len(set(r.label for r in results)) == 1

        self.results = results
        self._colorize = colorize
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = highlight_warnings
        self.label = results[0].label
        self.time_unit, self.time_scale = common.select_unit(
            min(r.median for r in results)
        )

        self.row_keys = common.ordered_unique([self.row_fn(i) for i in results])
        self.row_keys.sort(key=lambda args: args[:2])  # preserve stmt order
        self.column_keys = common.ordered_unique([self.col_fn(i) for i in results])
        self.rows, self.columns = self.populate_rows_and_columns()

    @staticmethod
    def row_fn(m: common.Measurement) -> Tuple[int, Optional[str], str]:
        return m.num_threads, m.env, m.as_row_name

    @staticmethod
    def col_fn(m: common.Measurement) -> Optional[str]:
        return m.description

    def populate_rows_and_columns(self) -> Tuple[Tuple[_Row, ...], Tuple[_Column, ...]]:
        rows: List[_Row] = []
        columns: List[_Column] = []
        ordered_results: List[List[Optional[common.Measurement]]] = [
            [None for _ in self.column_keys]
            for _ in self.row_keys
        ]
        row_position = {key: i for i, key in enumerate(self.row_keys)}
        col_position = {key: i for i, key in enumerate(self.column_keys)}
        for r in self.results:
            i = row_position[self.row_fn(r)]
            j = col_position[self.col_fn(r)]
            ordered_results[i][j] = r

        unique_envs = {r.env for r in self.results}
        render_env = len(unique_envs) > 1
        env_str_len = max(len(i) for i in unique_envs) if render_env else 0

        row_name_str_len = max(len(r.as_row_name) for r in self.results)

        prior_num_threads = -1
        prior_env = ""
        row_group = -1
        rows_by_group: List[List[List[Optional[common.Measurement]]]] = []
        for (num_threads, env, _), row in zip(self.row_keys, ordered_results):
            thread_transition = (num_threads != prior_num_threads)
            if thread_transition:
                prior_num_threads = num_threads
                prior_env = ""
                row_group += 1
                rows_by_group.append([])
            rows.append(
                _Row(
                    results=row,
                    row_group=row_group,
                    render_env=(render_env and env != prior_env),
                    env_str_len=env_str_len,
                    row_name_str_len=row_name_str_len,
                    time_scale=self.time_scale,
                    colorize=self._colorize,
                    num_threads=num_threads if thread_transition else None,
                )
            )
            rows_by_group[-1].append(row)
            prior_env = env

        for i in range(len(self.column_keys)):
            grouped_results = [tuple(row[i] for row in g) for g in rows_by_group]
            column = _Column(
                grouped_results=grouped_results,
                time_scale=self.time_scale,
                time_unit=self.time_unit,
                trim_significant_figures=self._trim_significant_figures,
                highlight_warnings=self._highlight_warnings,)
            columns.append(column)

        rows_tuple, columns_tuple = tuple(rows), tuple(columns)
        for ri in rows_tuple:
            ri.register_columns(columns_tuple)
        return rows_tuple, columns_tuple

    def render(self) -> str:
        string_rows = [[""] + self.column_keys]
        for r in self.rows:
            string_rows.append(r.as_column_strings())
        num_cols = max(len(i) for i in string_rows)
        for sr in string_rows:
            sr.extend(["" for _ in range(num_cols - len(sr))])

        col_widths = [max(len(j) for j in i) for i in zip(*string_rows)]
        finalized_columns = ["  |  ".join(i.center(w) for i, w in zip(string_rows[0], col_widths))]
        overall_width = len(finalized_columns[0])
        for string_row, row in zip(string_rows[1:], self.rows):
            finalized_columns.extend(row.row_separator(overall_width))
            finalized_columns.append("  |  ".join(row.finalize_column_strings(string_row, col_widths)))

        newline = "\n"
        has_warnings = self._highlight_warnings and any(ri.has_warnings for ri in self.results)
        return f"""
[{(' ' + (self.label or '') + ' ').center(overall_width - 2, '-')}]
{newline.join(finalized_columns)}

Times are in {common.unit_to_english(self.time_unit)}s ({self.time_unit}).
{'(! XX%) Measurement has high variance, where XX is the IQR / median * 100.' + newline if has_warnings else ""}"""[1:]


class Compare(object):
    def __init__(self, results: List[common.Measurement]):
        self._results: List[common.Measurement] = []
        self.extend_results(results)
        self._trim_significant_figures = False
        self._colorize = Colorize.NONE
        self._highlight_warnings = False

    def __str__(self):
        return "\n".join(self._render())

    def extend_results(self, results):
        for r in results:
            if not isinstance(r, common.Measurement):
                raise ValueError(
                    "Expected an instance of `Measurement`, " f"got {type(r)} instead."
                )
        self._results.extend(results)

    def trim_significant_figures(self):
        self._trim_significant_figures = True

    def colorize(self, rowwise=False):
        self._colorize = Colorize.ROWWISE if rowwise else Colorize.COLUMNWISE

    def highlight_warnings(self):
        self._highlight_warnings = True

    def print(self):
        print(str(self))

    def _render(self):
        results = common.Measurement.merge(self._results)
        grouped_results = self._group_by_label(results)
        output = []
        for group in grouped_results.values():
            output.append(self._layout(group))
        return output

    def _group_by_label(self, results: List[common.Measurement]):
        grouped_results: DefaultDict[str, List[common.Measurement]] = collections.defaultdict(list)
        for r in results:
            grouped_results[r.label].append(r)
        return grouped_results

    def _layout(self, results: List[common.Measurement]):
        table = Table(
            results,
            self._colorize,
            self._trim_significant_figures,
            self._highlight_warnings
        )
        return table.render()
