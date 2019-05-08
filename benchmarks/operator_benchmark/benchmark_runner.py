from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse

from caffe2.python import workspace

from operator_benchmark import benchmark_core, benchmark_utils

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
        '--run_mode',
        help='Run mode. '
        'short: run all operators with few shapes'
        'long: run all operators with all shapes',
        choices=benchmark_core.RUN_MODES.keys(),
        default='short')

    # This option is used to filter test cases to run.
    # Currently, the matching is sub-string but we can consider support regex.
    # For example, if test_case_filter = 'matmul', in will match these test
    # cases:
    # matmul_benchmark.Caffe2OperatorTestCase.matmul_512_128_512_transa_transb
    # matmul_benchmark.PyTorchOperatorTestCase.matmul_100_200_150
    # ...
    parser.add_argument(
        '--operator',
        help='Only run the test cases that contain the provided operator'
        ' as a substring of their names',
        default=None)

    parser.add_argument(
        '--list_tests',
        help='List all test cases without running them',
        action='store_true')

    parser.add_argument(
        "--iterations",
        help="Repeat each operator for the number of iterations",
        type=int
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
