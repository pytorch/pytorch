# Compare: Result visualization
## Overview

Robust benchmarking generally entails collecting a significant number of
measurements to measure different conditions. (shape, dtype, candidate impl, etc.)
Laying out these results in a comprehensible narative for author and reviewer
is a non-trivial task. `Compare` provides a modular and extensible way to layout
results from `Timer`.

## Semantics

`Compare` takes a list of `Union[Measurement, CallgrindStats]` in its constructor,
and displays them as a formatted table for easier analysis. Identical measurements
will be grouped, which allows `Compare` to process replicate measurements.

Grouping and layout is based on metadata passed to `Timer`:

* `label`: This is a top level description. (e.g. `add`, or `multiply`) one
table will be generated per unique label.

* `sub_label`: This is the row descriptor for a given configuration. Multiple
statements may be logically equivalent but differ in implementation. Assigning
separate sub_labels will result in a row per sub_label. If a sublabel is not
provided, `stmt` is used instead.

* `description`: This is the column descriptor for a given configuration. Unlike
`label` is does not have any intrinsic meaning and should be assigned on a
case-by-case basis as called for by the benchmarking task. (e.g. different input
shapes)

* `env`: An optional description of the torch environment. (e.g. `master` or
`my_branch`). Like sub_labels, statistics are calculated across envs. (Since
comparing a branch to master or a stable release is a common use case.)
However `Compare` will visually group rows which are run with the same `env`.

* `num_threads`: By default, `Timer` will run in single-threaded mode. If
`Measurements` with different numbers of threads are given to `Compare`, they
will be grouped into separate blocks of rows.

`Compare` also offers several configuration options as constructor arguments:
* `data_cell`: See the `Advanced` section.
* `transpose`: Use `description` as the row key and `sub_label` as the column key.
* `rowwise`: Use rows, rather than columns when grouping values for analysis.
* `colorize`: Highlight variation by coloring elements based on their relative
performance compared to other cells in a column. (Or row if `rowwise=True`.)
* `trim_significant_figures`: Vary the displayed precision depending on how
much variation is present in measurements.
* `highlight_warnings`: Annotate measurements with particularly high variation,
as these may not be reliable.


## Advanced

While `Compare` aims to provide a good out-of-the-box experience, one may wish
to analyze the data in a task-specific way. To facilitate this, `Compare`
accepts a `data_cell` argument which allows users to override the default render
behavior. The base class takes as constructor arguments the wall times and
instruction counts associated with a particular cell, as well as a config object
which packages `Compare`'s constructor arguments. (`rowwise`, `colorize`, etc.)

```
class DataCellBase(Cell):
    def __init__(
        self,
        wall_times: Optional[Tuple[common.Measurement, ...]],
        instruction_counts: Optional[Tuple[timer_interface.CallgrindStats, ...]],
        config: DataCellConfig,
    ) -> None:
        # Your code here.
        ...

    def render(self) -> str:
        # Your code here.
        ...
```

A subclassed data cell may perform arbitrary analysis so long as it implements
the `render` method. DataCell implementers do not need to worry about minutuae
such as padding and alignment; the enclosing `Table` will handle that. (Though
it is possible to excercise finer control with the `alignment` property.)

However the contents of a cell might depend no only on that cell's values, but
also its neighbors. For instance, when selecting a unit (ms, us, etc.) it would
be jarring for each cell to select a separate unit; the choice should be chosen
based on all the elements in the table. `Cell` exposes three methods for this:
* `row_reduce(...)`
* `col_reduce(...)`
* `all_reduce(...)`

These methods provide an interface for a cell to access all other cells in a row,
column, or entire Table and implement more advanced logic. They also handle
caching, so cells can simply call the reduction methods and not have to worry
about run time for large tables. These methods will be more clear with an example.


## Example:

It is well known that activations tend to be memory bound. Let's demonstrate
this phenomenon:


```
import torch

from torch.utils.benchmark import Timer
from torch.utils.benchmark import Compare

results = []
for k in (10, 14, 16, 18, 20, 22, 24, 26):
    for dtype in (torch.float32, torch.int8):
        measurement = Timer(
            "x.relu()",
            globals={"x": torch.zeros((2 ** k,), dtype=dtype)},
            label="ReLU",
            sub_label=f"size: 2 ** {k}",
            description=f"{dtype}",
        ).blocked_autorange()
        measurement.metadata = {"dtype": dtype, "size": 2 ** k}
        results.append(measurement)

Compare(results).print()
```

```
[----------------- ReLU -----------------]
              | torch.float32 | torch.int8
------------------------------------------
size: 2 ** 10 |          2.5  |       2.4
size: 2 ** 14 |          5.7  |       3.2
size: 2 ** 16 |         26.0  |       6.3
size: 2 ** 18 |         93.7  |      26.0
size: 2 ** 20 |        355.7  |      92.1
size: 2 ** 22 |       3139.1  |     400.2
size: 2 ** 24 |      52286.5  |    4743.7
size: 2 ** 26 |     214376.0  |   52509.3

Times are in microseconds (us).
```

ReLU takes longer with more and larger elements, but there are more subtle
effects in play. Let's use bandwidth to investigate.

```
from torch.utils.benchmark import DataCellBase, DataCellConfig

class BandwidthDataCell(DataCellBase):
    def __init__(
        self,
        wall_times,

        # We are not interested in instruction counts of table options in this example.
        **kwargs
    ) -> None:
        assert len(wall_times) == 1
        self.wall_time = wall_times[0]

    @property
    def bandwidth(self):
        n_gigabytes = (
            self.wall_time.metadata["size"] *
            {torch.float32: 4, torch.int8: 1}[self.wall_time.metadata["dtype"]]
        ) / 1024 ** 3

        return n_gigabytes / self.wall_time.median, n_gigabytes / min(self.wall_time.times)

    def render(self) -> str:
        median_bandwidth, _ = self.bandwidth

        # Use the best performing cell in the table as a proxy for system capacity.
        # We use `all_reduce` to get this data.
        peak_bandwidth, _ = self.all_reduce(self.get_peak)
        ratio = median_bandwidth / peak_bandwidth

        result = f"{median_bandwidth:>5.1f} | {ratio * 100:>4.0f} %"

        # We can use `col_reduce` to conditionally render the last cell in a column.
        unit, cell_position = self.col_reduce(self.get_unit)

        if unit[cell_position]:
            result = f"{result}\n{'-' * len(result)}\n{unit[cell_position]:>5} | % peak"
        return result

    @staticmethod
    def get_peak(cells):
        return max([
            cell.bandwidth[1] for cell in cells
            if isinstance(cell, BandwidthDataCell)
        ])

    @staticmethod
    def get_unit(cells):
        result = [None for _ in cells]
        index = max([i for i, cell in enumerate(cells) if isinstance(cell, BandwidthDataCell)])
        result[index] = "GB/s"
        return tuple(result)

Compare(results, data_cell=BandwidthDataCell).print()
```

```
[-------------------- ReLU -------------------]
              |  torch.float32 |     torch.int8
-----------------------------------------------
size: 2 ** 10 |   1.5 |   13 % |   0.4 |    3 %
size: 2 ** 14 |  10.7 |   93 % |   4.8 |   42 %
size: 2 ** 16 |   9.4 |   82 % |   9.7 |   84 %
size: 2 ** 18 |  10.4 |   91 % |   9.4 |   82 %
size: 2 ** 20 |  11.0 |   96 % |  10.6 |   92 %
size: 2 ** 22 |   5.0 |   43 % |   9.8 |   85 %
size: 2 ** 24 |   1.2 |   10 % |   3.3 |   29 %
size: 2 ** 26 |   1.2 |   10 % |   1.2 |   10 %
              | -------------- | --------------
              |  GB/s | % peak |  GB/s | % peak
```

This clearly highlights the sweet spot of activation sizes: if the Tensor is too
small we are overhead bound, if it is too large we spill out of cache and become
memory bound.
