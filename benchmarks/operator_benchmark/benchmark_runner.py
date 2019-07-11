from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse

from caffe2.python import workspace

import benchmark_core
import benchmark_utils

"""Performance microbenchmarks's main binary.

This is the main function for running performance microbenchmark tests.
It also registers existing benchmark tests via Python module imports.
"""


def main():
    print("Python version " + str(sys.version_info[0]))

    parser = argparse.ArgumentParser(
        description="Run microbenchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--tag_filter',
        help='tag_filter can be used to run the benchmarks which matches the tag',
        default='short')

    # This option is used to filter test cases to run.
    parser.add_argument(
        '--operator',
        help='Run the test cases that contain the provided operator'
        ' as a substring of their names',
        default=None)

    parser.add_argument(
        '--test_name',
        help='Run tests that have the provided test_name',
        default=None)

    parser.add_argument(
        '--list_ops',
        help='List all test cases without running them',
        action='store_true')

    parser.add_argument(
        "--iterations",
        help="Repeat each operator for the number of iterations",
        type=int
    )

    parser.add_argument(
        "--min_time_per_test",
        help="Set the minimum time (unit: seconds) to run each test",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--warmup_iterations",
        help="Number of iterations to ignore before measuring performance",
        default=10,
        type=int
    )

    parser.add_argument(
        "--ai_pep_format",
        help="Print result when running on AI-PEP",
        default=False,
        type=bool
    )

    parser.add_argument(
        "--forward_only",
        help="Only run the forward path of operators",
        action='store_true'
    )

    parser.add_argument(
        '--framework',
        help='Comma-delimited list of frameworks to test (Caffe2, PyTorch)',
        default="Caffe2,PyTorch")

    args = parser.parse_args()

    if benchmark_utils.is_caffe2_enabled(args.framework):
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        workspace.ClearGlobalNetObserver()

    benchmark_core.BenchmarkRunner(args).run()


if __name__ == "__main__":
    main()
