from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from operator_benchmark import benchmark_core

import torch

"""PyTorch performance microbenchmarks.

This module contains PyTorch-specific functionalities for performance
microbenchmarks.
"""


def PyTorchOperatorTestCase(test_name, op_type, input_shapes, op_args, run_mode):
    """Benchmark Tester function for Pytorch framework.
    """
    inputs = []
    is_contig = 'contig' not in op_args or op_args['contig']
    dtype = op_args['dtype'] if 'dtype' in op_args else torch.float32
    for shape in input_shapes:
        tensor_shape = list(shape)
        if not is_contig:
            tensor_shape = [s * 2 for s in tensor_shape]
        if dtype in [torch.float32, torch.float64]:
            input = torch.rand(tensor_shape, dtype=dtype)
        elif dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            input = torch.randint(low=0, high=100, size=tensor_shape, dtype=dtype)
        else:
            input = torch.ones(tensor_shape, dtype=dtype)

        if not is_contig:
            slices = []
            for dim in tensor_shape:
                slices.append(slice(0, dim, 2))
            input = input[slices]
            assert list(input.size()) == list(shape)
            assert not input.is_contiguous()
        inputs.append(input)

    def benchmark_func(num_runs):
        op_type(*(inputs + [num_runs]))

    benchmark_core.add_benchmark_tester("PyTorch", test_name, input_shapes, op_args, run_mode, benchmark_func)
