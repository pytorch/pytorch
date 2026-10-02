# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kineto trace analysis benchmark.
"""

import argparse
import sys

from .backends import AVAILABLE_TASKS, DEFAULT_METRICS
from .framework import TraceAnalysisBenchmark


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["torchbench_resnet50_3080ti"],
        help="Name of the inputs.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=AVAILABLE_TASKS, help="Name of the tasks."
    )
    parser.add_argument(
        "--backends", nargs="+", default=["perfetto"], help="Name of the backends."
    )
    parser.add_argument(
        "--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to collect."
    )

    parser.add_argument("--csv", action="store_true", help="Output the result as csv")
    parser.add_argument(
        "--warmup", default=10, type=int, help="Number of warmup iterations."
    )

    parser.add_argument("--iter", default=20, type=int, help="Run iterations.")
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    benchmark = TraceAnalysisBenchmark(args)
    benchmark.run()
    result = benchmark.result

    if args.csv:
        print(result.write_csv_to_file(sys.stdout))
    else:
        print(result)
