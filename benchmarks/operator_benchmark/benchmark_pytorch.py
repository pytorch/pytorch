from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.benchmarks.operator_benchmark import benchmark_core, benchmark_utils

import torch

"""PyTorch performance microbenchmarks.

This module contains PyTorch-specific functionalities for performance
microbenchmarks.
"""


def PyTorchOperatorTestCase(test_name, op_type, input_shapes, op_args, run_mode):
    """Benchmark Tester function for Pytorch framework.
    test_case is expected to be a PyTorchOperatorTestCase object. If not, the
    function will return False.
    It returns a function that contains the code to benchmarked
    (operator execution).
    """
    inputs = [torch.from_numpy(benchmark_utils.numpy_random_fp32(*input)) for input in input_shapes]

    def benchmark_func(num_runs):
        op_type(*(inputs + [num_runs]))

    benchmark_core.add_benchmark_tester("PyTorch", test_name, input_shapes, op_args, run_mode, benchmark_func)
