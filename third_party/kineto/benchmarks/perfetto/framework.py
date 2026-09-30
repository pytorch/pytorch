# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from .backends import AVAILABLE_BACKENDS, DEFAULT_METRICS
from .table import TraceAnalysisBenchmarkResult


class TraceAnalysisBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.inputs = args.inputs
        self.tasks = args.tasks
        self.backends = {
            x_val(args)
            for x_name, x_val in AVAILABLE_BACKENDS.items()
            if x_name in args.backends
        }

        self.metrics = args.metrics if args.metrics else DEFAULT_METRICS

        assert self.inputs, "Inputs to benchmark cannot be empty."
        assert self.tasks, "Tasks to benchmark cannot be empty."
        assert self.backends, "Backends to benchmark cannot be empty."

        self.result = TraceAnalysisBenchmarkResult(
            inputs=self.inputs,
            tasks=self.tasks,
            metrics=self.metrics,
        )

    def run(self):
        for backend in self.backends:
            for input in self.inputs:
                backend._load(input)
                for task in filter(lambda x: not x == "load", self.tasks):
                    backend._run(task)
            result_key = (input, backend.name)
            self.result.data[result_key] = backend.output
