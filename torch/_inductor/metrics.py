from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache

from typing import List, Set, Tuple, TYPE_CHECKING, Union

from torch._inductor import config
from torch._inductor.utils import get_benchmark_name

# Prevent circular import
if TYPE_CHECKING:
    from torch._inductor.scheduler import (
        BaseSchedulerNode,
        ExternKernelSchedulerNode,
        NopKernelSchedulerNode,
        SchedulerNode,
    )

# counter for tracking how many kernels have been generated
generated_kernel_count = 0
generated_cpp_vec_kernel_count = 0
num_bytes_accessed = 0
nodes_num_elem: List[
    Tuple[
        Union[NopKernelSchedulerNode, SchedulerNode, ExternKernelSchedulerNode],
        int,
    ]
] = []
node_runtimes: List[Tuple[BaseSchedulerNode, float]] = []

# counters for tracking fusions
ir_nodes_pre_fusion = 0

# counters for tracking to_dtype inserted
cpp_to_dtype_count = 0

# counters for tracking cpp_wrapper disabled
disable_cpp_wrapper = 0


# reset all counters
def reset():
    global generated_kernel_count
    global generated_cpp_vec_kernel_count
    global num_bytes_accessed, nodes_num_elem
    global ir_nodes_pre_fusion
    global cpp_to_dtype_count
    global disable_cpp_wrapper

    generated_kernel_count = 0
    generated_cpp_vec_kernel_count = 0
    num_bytes_accessed = 0
    nodes_num_elem.clear()
    node_runtimes.clear()
    ir_nodes_pre_fusion = 0
    cpp_to_dtype_count = 0
    disable_cpp_wrapper = 0


REGISTERED_METRIC_TABLES = {}


@dataclass
class MetricTable:
    table_name: str
    column_names: List[str]

    num_rows_added: int = 0

    def add_row(self, row_fn):
        if self.table_name not in enabled_metric_tables():
            return

        row_dict = row_fn()
        assert len(self.column_names) == len(
            row_dict
        ), f"{len(self.column_names)} v.s. {len(row_dict)}"
        assert set(self.column_names) == set(
            row_dict.keys()
        ), f"{set(self.column_names)} v.s. {set(row_dict.keys())}"

        row = [
            get_benchmark_name(),
        ]
        row += [row_dict[column_name] for column_name in self.column_names]
        self._write_row(row)

    def output_filename(self):
        return f"metric_table_{self.table_name}.csv"

    def write_header(self):
        filename = self.output_filename()
        with open(filename, "w") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(["model_name"] + self.column_names)

    def _write_row(self, row):
        filename = self.output_filename()
        if self.num_rows_added == 0 and not os.path.exists(filename):
            self.write_header()

        self.num_rows_added += 1

        for idx, orig_val in enumerate(row):
            if isinstance(orig_val, float):
                new_val = f"{orig_val:.6f}"
            elif orig_val is None:
                new_val = ""
            else:
                new_val = orig_val
            row[idx] = new_val

        with open(filename, "a") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(row)

    @staticmethod
    def register_table(name, column_names):
        table = MetricTable(name, column_names)
        REGISTERED_METRIC_TABLES[name] = table


MetricTable.register_table(
    "slow_fusion",
    [
        "kernel1_path",
        "kernel1_latency",
        "kernel2_path",
        "kernel2_latency",
        "fused_kernel_path",
        "fused_kernel_latency",
        "slow_down_ratio",
    ],
)

# track the fusion statistics for each graph
MetricTable.register_table(
    "graph_stats",
    [
        "graph_id",
        "num_nodes_before_fusion",
        "num_nodes_after_fusion",
    ],
)


def purge_old_log_files():
    """
    Purge the old log file at the beginning when the benchmark script runs.
    Should do it in the parent process rather than the child processes running
    each individual model.
    """
    for name, table in REGISTERED_METRIC_TABLES.items():
        if name in enabled_metric_tables():
            filename = table.output_filename()
            if os.path.exists(filename):
                os.unlink(filename)

            table.write_header()


@lru_cache
def enabled_metric_tables() -> Set[str]:
    config_str = config.enabled_metric_tables

    enabled = set()
    for name in config_str.split(","):
        name = name.strip()
        if not name:
            continue
        assert (
            name in REGISTERED_METRIC_TABLES
        ), f"Metric table name {name} is not registered"
        enabled.add(name)
    return enabled


def is_metric_table_enabled(name):
    return name in enabled_metric_tables()


def get_metric_table(name):
    assert name in REGISTERED_METRIC_TABLES, f"Metric table {name} is not defined"
    return REGISTERED_METRIC_TABLES[name]
