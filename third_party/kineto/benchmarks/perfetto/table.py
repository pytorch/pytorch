# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import tabulate

from .backends.common import TraceAnalysisMetrics


@dataclass
class TraceAnalysisBenchmarkResult:
    inputs: List[str]
    tasks: List[str]
    metrics: List[str]
    # key: (input, backend), value: benchmark results by tasks
    data: Dict[Tuple[str, str], TraceAnalysisMetrics] = field(default_factory=dict)

    def _table(self):
        """
        Generate the output table.
        Row: input-task
        Column: backend-metric
        """
        table = []
        # generate headers
        headers = ["input-task"]
        headers.extend(
            [
                f"{backend}-{metric}"
                for metric in self.metrics
                for (_input, backend) in self.data.keys()
            ]
        )

        if len(self.data.items()) == 0:
            return headers, table
        # generate table rows
        for input in self.inputs:
            for task in self.tasks:
                row_table = [f"{input}-{task}"]
                for key in filter(lambda x: input in x, self.data.keys()):
                    metric_dict = asdict(self.data[key])
                    for metric in self.metrics:
                        row_table.append(metric_dict.get(metric).get(task, None))
                table.append(row_table)
        return headers, table

    def write_csv_to_file(self, fileobj):
        import csv

        headers, table = self._table()
        writer = csv.writer(fileobj, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(table)

    def write_csv(self, dir_path):
        import tempfile

        # This is just a way to create a unique filename. It's not actually a
        # temporary file (since delete=False).
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=os.path.join(dir_path, "perfettobenchmark"),
            suffix=".csv",
            newline="",
            delete=False,
        ) as fileobj:
            self.write_csv_to_file(fileobj)
            return fileobj.name

    def _get_result_dict(self):
        return self.data

    def __str__(self):
        headers, table = self._table()
        table = tabulate.tabulate(table, headers=headers, stralign="right")
        return table
