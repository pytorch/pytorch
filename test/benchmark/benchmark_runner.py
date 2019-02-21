from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import argparse

from caffe2.python import workspace

import caffe2.test.benchmark.benchmark_core as bc
import caffe2.test.benchmark.benchmark_caffe2
import caffe2.test.benchmark.benchmark_pytorch

import caffe2.test.benchmark.ops.add_benchmark
import caffe2.test.benchmark.ops.matmul_benchmark

"""Performance microbenchmarks's main binary.

This is the main function for running performance microbenchmark tests.
It also registers existing benchmark tests via Python module imports.
"""


if __name__ == "__main__":
    print("Python version " + str(sys.version_info[0]))

    parser = argparse.ArgumentParser(
        description="Run microbenchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--run_mode',
        help='Run mode. '
        'In long mode, each microbenchmark test is repeated more times and long'
        ' running tests are executed',
        choices=bc.RUN_MODES.keys(),
        default='short')

    # This option is used to filter test cases to run.
    # Currently, the matching is sub-string but we can consider support regex.
    # For example, if test_case_filter = 'matmul', in will match these test
    # cases:
    # matmul_benchmark.Caffe2OperatorTestCase.matmul_512_128_512_transa_transb
    # matmul_benchmark.PyTorchOperatorTestCase.matmul_100_200_150
    # ...
    parser.add_argument(
        '--test_case_filter',
        help='Only run the test cases that contain the provided filter'
        ' as a substring of their names',
        default=None)

    parser.add_argument(
        '--list_tests',
        help='List all test cases without running them',
        action='store_true')

    args = parser.parse_args()

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])

    # Reuse test_case_filter as a way of listing test cases
    test_case_filter = 'NO_TEST' if args.list_tests else args.test_case_filter
    bc.BenchmarkRunner(args.run_mode, test_case_filter).run()
