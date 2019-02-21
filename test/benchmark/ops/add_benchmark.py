from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import caffe2.test.benchmark.benchmark_core as bc
import caffe2.test.benchmark.benchmark_utils as bu
from caffe2.test.benchmark.benchmark_caffe2 import Caffe2OperatorTestCase
from caffe2.test.benchmark.benchmark_pytorch import PyTorchOperatorTestCase

import torch


"""Microbenchmarks for element-wise Add operator. Supports both Caffe2/PyTorch."""

# Input shapes that we test and the run mode for each shape.
TEST_SHAPES = [
    ([1000], 'short'), ([512, 512], 'short'), ([2000, 1024, 200], 'long')
]


def torch_add(a, b):
    """A simple wrapper for torch.add that accepts two tensors as the second
    argument of torch.add is the scalar multiplier.
    """
    return torch.add(a, 1, b)


def add_test_cases(shape, run_mode):
    test_name = 'add_%s' % bu.shape_to_string(shape)
    input_shapes = [shape, shape]
    result = [Caffe2OperatorTestCase(
        test_name=test_name,
        op_type='Add',
        input_shapes=input_shapes,
        op_args={},
        run_mode=run_mode)]
    result.append(PyTorchOperatorTestCase(
        test_name=test_name,
        op_type=torch_add,
        input_shapes=input_shapes,
        op_args={},
        run_mode=run_mode))
    return result


@bc.benchmark_test
def add_benchmark():
    result = []
    for test_case in TEST_SHAPES:
        result.extend(add_test_cases(test_case[0], test_case[1]))
    return result
