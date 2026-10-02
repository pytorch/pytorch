# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time
from dataclasses import dataclass, field, fields
from typing import Callable, Dict

import numpy

from .. import BENCHMARK_DATA_DIR


def _get_input_path(input_name):
    input_name = f"{input_name}.json"
    return os.path.join(BENCHMARK_DATA_DIR, "torchbench_traces", input_name)


@dataclass
class TraceAnalysisMetrics:
    # Latency to perform trace analysis tasks
    latency: Dict[str, float] = field(default_factory=dict)
    # Peak CPU memory to perform trace analysis tasks
    peak_mem: Dict[str, float] = field(default_factory=dict)
    # extra metrics
    extra_metrics: Dict[str, float] = field(default_factory=dict)


DEFAULT_METRICS = ["latency"]
BUILTIN_METRICS = {x.name for x in fields(TraceAnalysisMetrics)} - {"extra_metrics"}


class TraceAnalysis:
    output: TraceAnalysisMetrics

    def __init__(self, args: argparse.Namespace):
        self.output = TraceAnalysisMetrics()
        self.warmup = args.warmup
        self.iter = args.iter

    def _load(self, input: str):
        input_path = _get_input_path(input)
        t_iter_begin = time.perf_counter()
        self.load(input_path)
        t_iter_end = time.perf_counter()
        self.output.latency["load"] = t_iter_end - t_iter_begin

    def _run(self, task: str):
        run_lambda = self.run(task)
        # warmup
        for _ in range(self.warmup):
            run_lambda()
        latencies = []
        # TODO: does perfetto cache the query result?
        for _ in range(self.iter):
            t_iter_begin = time.perf_counter()
            run_lambda()
            t_iter_end = time.perf_counter()
            latencies.append(t_iter_end - t_iter_begin)
        # record p50 latency only
        self.output.latency[task] = numpy.median(latencies)

    def load(self, input_file_path: str):
        raise NotImplementedError("Trace loading is not implemented yet.")

    def run(self, task: str) -> Callable:
        task_lambda = getattr(self, task, None)
        if not task_lambda:
            raise NotImplementedError(f"Task {task} is not implemented yet.")
        return lambda: task_lambda()
