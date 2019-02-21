from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import numpy as np

import caffe2.test.benchmark.benchmark_core as bc
import caffe2.test.benchmark.benchmark_utils as bu

import torch

"""PyTorch performance microbenchmarks.

This module contains PyTorch-specific functionalities for performance
microbenchmarks.
"""


PyTorchOperatorTestCase = namedtuple(
    "PyTorchOperatorTestCase",
    ["test_name", "op_type", "input_shapes", "op_args", "run_mode"])


@bc.benchmark_tester
def pytorch_tester(test_case):
    """Benchmark Tester function for Pytorch framework.
    test_case is expected to be a PyTorchOperatorTestCase object. If not, the
    function will return False.
    It returns a function that contains the code to benchmarked
    (operator execution).
    """
    if type(test_case) is PyTorchOperatorTestCase:
        print("Running benchmark test case %s with pytorch" % (test_case.test_name))

        inputs = [torch.from_numpy(bu.numpy_random_fp32(*input)) for input in test_case.input_shapes]

        def benchmark_func():
            test_case.op_type(*inputs)

        return benchmark_func
    return False
