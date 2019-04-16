from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.benchmarks.operator_benchmark import benchmark_core, benchmark_utils

from caffe2.benchmarks.operator_benchmark.benchmark_caffe2 import Caffe2OperatorTestCase
from caffe2.benchmarks.operator_benchmark.benchmark_pytorch import PyTorchOperatorTestCase

import torch


"""Microbenchmarks for element-wise Add operator. Supports both Caffe2/PyTorch."""

# Input shapes that we test and the run mode for each shape.
# Sum up two tensors with the same shape


def generate_inputs():
    ms = benchmark_utils.get_n_rand_nums(min_val=1, max_val=128, n=1)
    ns = benchmark_utils.get_n_rand_nums(min_val=1, max_val=128, n=2)
    ks = benchmark_utils.get_n_rand_nums(min_val=1, max_val=128, n=2)
    mode = ['long']

    test_cases = benchmark_utils.cross_product([ms], mode)

    two_dims = benchmark_utils.cross_product(ms, ns)
    two_dims = benchmark_utils.cross_product(two_dims, mode)
    test_cases.extend(two_dims)

    three_dims = benchmark_utils.cross_product(ms, ns, ks)
    three_dims = benchmark_utils.cross_product(three_dims, mode)
    test_cases.extend(three_dims)

    # Representative inputs
    test_cases.extend([([128], 'short'),
                       ([64, 128], 'short'),
                       ([32, 64, 128], 'short')])
    return test_cases


@torch.jit.script
def torch_add(a, b, iterations):
    # type: (Tensor, Tensor, int)
    result = torch.jit.annotate(torch.Tensor, None)
    for _ in range(iterations):
        result = torch.add(a, b)
    return result


@benchmark_core.benchmark_test_group
def add_test_cases():
    test_cases = generate_inputs()
    for test_case in test_cases:
        X, run_mode = test_case
        Caffe2OperatorTestCase(
            test_name='add',
            op_type='Add',
            input_shapes=[X, X],
            op_args={},
            run_mode=run_mode)
        PyTorchOperatorTestCase(
            test_name='add',
            op_type=torch_add,
            input_shapes=[X, X],
            op_args={},
            run_mode=run_mode)
