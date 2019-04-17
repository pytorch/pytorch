from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.benchmarks.operator_benchmark import benchmark_core, benchmark_utils

from caffe2.benchmarks.operator_benchmark.benchmark_caffe2 import Caffe2OperatorTestCase
from caffe2.benchmarks.operator_benchmark.benchmark_pytorch import PyTorchOperatorTestCase

import torch


"""Microbenchmarks for MatMul operator. Supports both Caffe2/PyTorch."""


def generate_inputs():
    # Random inputs
    Ms = benchmark_utils.get_n_rand_nums(min_val=1, max_val=128, n=2)
    Ns = benchmark_utils.get_n_rand_nums(min_val=1, max_val=128, n=2)
    Ks = benchmark_utils.get_n_rand_nums(min_val=1, max_val=128, n=2)
    transpose_a = [False, True]
    transpose_b = [True, False]
    mode = ['long']
    test_cases = benchmark_utils.cross_product(Ms, Ns, Ks, transpose_a, transpose_b, mode)

    # Representative inputs
    test_cases.extend([(8, 16, 64, False, False, 'short'),
                       (64, 64, 256, False, False, 'short'),
                       (256, 256, 256, False, False, 'short')])
    return test_cases


@torch.jit.script
def torch_matmul(a, b, iterations):
    # type: (Tensor, Tensor, int)
    result = torch.jit.annotate(torch.Tensor, None)
    for _ in range(iterations):
        result = torch.matmul(a, b)
    return result


@benchmark_core.benchmark_test_group
def matmul_test_cases():
    test_cases = generate_inputs()
    for test_case in test_cases:
        M, N, K, trans_a, trans_b, run_mode = test_case
        input_shapes = [(N, M) if trans_a else (M, N), (K, N) if trans_b else (N, K)]
        Caffe2OperatorTestCase(
            test_name='matmul',
            op_type='MatMul',
            input_shapes=input_shapes,
            op_args={'trans_a': trans_a, 'trans_b': trans_b},
            run_mode=run_mode)
        if not trans_a and not trans_b:
            # PyTorch's matmul does not take transpose flags, so we only
            # have a test case when there are no transpose flags.
            PyTorchOperatorTestCase(
                test_name='matmul',
                op_type=torch_matmul,
                input_shapes=input_shapes,
                op_args={},
                run_mode=run_mode)
