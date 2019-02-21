from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import caffe2.test.benchmark.benchmark_core as bc
from caffe2.test.benchmark.benchmark_caffe2 import Caffe2OperatorTestCase
from caffe2.test.benchmark.benchmark_pytorch import PyTorchOperatorTestCase

import torch


"""Microbenchmarks for MatMul operator. Supports both Caffe2/PyTorch."""


# Input shapes that we test and the run mode for each shape.
TEST_SHAPES = [
    (100, 200, 150, 'short'), (512, 128, 512, 'short'), (2000, 1000, 3000, 'long')
]


def matmul_test_cases(M, N, K, trans_a, trans_b, run_mode):
    input_shapes = [(N, M) if trans_a else (M, N), (K, N) if trans_b else (N, K)]
    test_name = 'matmul_%d_%d_%d' % (M, N, K)
    if trans_a:
        test_name = test_name + "_transa"
    if trans_b:
        test_name = test_name + "_transb"
    result = [Caffe2OperatorTestCase(
        test_name=test_name,
        op_type='MatMul',
        input_shapes=input_shapes,
        op_args={'trans_a': trans_a, 'trans_b': trans_b},
        run_mode=run_mode)]
    if not trans_a and not trans_b:
        # PyTorch's matmul does not take transpose flags, so we only
        # have a test case when there are no transpose flags.
        result.append(PyTorchOperatorTestCase(
            test_name=test_name,
            op_type=torch.matmul,
            input_shapes=input_shapes,
            op_args={},
            run_mode=run_mode))
    return result


@bc.benchmark_test
def matmul_benchmark():
    result = []
    for transpose_args in [(False, False), (False, True), (True, True)]:
        for shape in TEST_SHAPES:
            result.extend(matmul_test_cases(
                shape[0], shape[1], shape[2], transpose_args[0], transpose_args[1], run_mode=shape[3]))
    return result
