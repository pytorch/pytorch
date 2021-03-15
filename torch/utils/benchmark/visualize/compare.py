import collections
import itertools as it
from typing import Any, Dict, List, Optional, Type, Union

from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface
from torch.utils.benchmark.visualize import cells
from torch.utils.benchmark.visualize import table

__all__ = ["Compare"]


class Compare:
    def __init__(
        self,
        results: table.RESULTS,
        data_cell: Type[cells.DataCellBase] = cells.DataCell,
        *,
        transpose: bool = False,
        rowwise: bool = False,
        colorize: bool = False,
        trim_significant_figures: bool = False,
        highlight_warnings: bool = False
    ) -> None:
        self._results = results
        self._data_cell = data_cell

        self._transpose: bool = transpose
        self._cell_config = cells.DataCellConfig(
            rowwise=rowwise,
            colorize=colorize,
            trim_significant_figures=trim_significant_figures,
            highlight_warnings=highlight_warnings,
        )

    def __str__(self) -> str:
        return self._render()

    def print(self) -> None:
        print(str(self))

    def _render(self) -> str:
        results_by_label = collections.defaultdict(list)
        for r in self._results:
            results_by_label[r.task_spec.label].append(r)

        tables = tuple(
            self._assemble_table(label, segment)
            for label, segment in results_by_label.items()
        )

        return "\n\n".join([table.render() for table in tables])

    def _assemble_table(self, label: Optional[str], result_segment: table.RESULTS) -> table.Table:
        result_groups = collections.defaultdict(list)
        for r in result_segment:
            row_key = r.task_spec.as_row_name
            col_key = r.task_spec.description or ""
            if self._transpose:
                row_key, col_key = col_key, row_key
            result_groups[(r.task_spec.num_threads, r.task_spec.env)].append((row_key, col_key, r))

        row_keys = common.ordered_unique([k for k, _, _ in it.chain(*result_groups.values())])
        col_keys = common.ordered_unique([k for _, k, _ in it.chain(*result_groups.values())])

        # MyPy gets confused if we don't type annotate before adding.
        null_corner: List[table.Cell] = [cells.NullCell()]

        table_rows: List[Union[table.FullRowCell, List[table.Cell]]] = [
            cells.TitleHeader(title=label),
            null_corner + [cells.ColHeader(col_key) for col_key in col_keys],
            cells.FallbackHeader(),
        ]

        for num_threads, env in sorted(result_groups.keys(), key=lambda x: x[0]):
            table_rows.append(cells.NumThreadsHeader(n=num_threads))
            sub_table: Dict[str, Dict[str, List[table.RESULT]]] = {
                row_key: {col_key: [] for col_key in col_keys}
                for row_key in row_keys
            }

            for row_key, col_key, r in result_groups[(num_threads, env)]:
                sub_table[row_key][col_key].append(r)

            for row_key, row in sub_table.items():
                table_rows.append([cells.RowHeader(env=env, row_key=row_key)])
                for col_key in col_keys:
                    wall_times: List[common.Measurement] = []
                    instruction_counts: List[timer_interface.CallgrindStats] = []
                    for r in row[col_key]:
                        if isinstance(r, common.Measurement):
                            wall_times.append(r)
                        else:
                            assert isinstance(r, timer_interface.CallgrindStats)
                            instruction_counts.append(r)

                    cell = self._data_cell(
                        wall_times=tuple(wall_times) or None,
                        instruction_counts=tuple(instruction_counts) or None,
                        config=self._cell_config,
                    )

                    # Narrow the Union for MyPy
                    assert isinstance(table_rows[-1], list)
                    table_rows[-1].append(cell)
        table_rows.append(cells.FooterCell(self._data_cell))

        return table.Table(table_rows)

    def extend_results(self, results: Any) -> None:
        raise NotImplementedError("`extend_results` has been removed from Compare.")

    def trim_significant_figures(self) -> None:
        raise NotImplementedError("`trim_significant_figures` is now a constructor arg.")

    def colorize(self, rowwise: bool = False) -> None:
        raise NotImplementedError("`colorize` and `rowwise` are now a constructor args.")

    def highlight_warnings(self) -> None:
        raise NotImplementedError("`highlight_warnings` is now a constructor arg.")
