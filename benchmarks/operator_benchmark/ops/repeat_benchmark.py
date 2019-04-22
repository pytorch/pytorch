from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from benchmarks.operator_benchmark import benchmark_core, benchmark_runner
from benchmarks.operator_benchmark.benchmark_test_generator import *
from benchmarks.operator_benchmark import benchmark_utils

from functools import reduce

import torch

from enum import Enum

"""Microbenchmarks for Tensor repeat operator. Supports PyTorch."""

class DType(Enum):
    float = 1

# Short config
short_config = generate_configs(
    M=[2],
    N=[1],
    mode=['short'],
    sample_func=cross_product
)

input_shapes = (
               (4, 4, 1),
               (16, 1, 32),
               (64, 64, 1, 1),
               (8, 256, 128),
               (1, 64, 128, 32),
               (512, 512),
               )

repeats= (
         (1, 1, 1, 64),
         (1, 4, 1, 2),
         (1, 2, 2, 15),
         (1, 1, 3, 2),
         (128, 1, 8, 1),
         (1, 1, 2, 16),
         )

def generate_data_for_repeat(dtype=DType.float):
    inputs = [torch.from_numpy(benchmark_utils.numpy_random_fp32(*input)) for input in input_shapes]
    total_num_elements = 0
    for i, input in enumerate(inputs):
        total_num_elements += input.numel()
        total_num_elements += input.numel() * reduce(lambda x, y: x*y, repeats[i])
    return inputs, (total_num_elements * 4)

inputs, total_bytes = generate_data_for_repeat()
BYTES_TO_MB = (1./1000./1000.)

def map_dims_to_shape(M, N):
    in_shape = (M, N)
    in_shapes = [in_shape]
    args = {}
    return (in_shapes, args)

def torch_repeat(input_tensor, repeat):
    return input_tensor.repeat(repeat)

def torch_repeat_dummy(a, iterations):
    result = []
    for i, input in enumerate(inputs):
        result.append(torch_repeat(input,repeats[i]))
    return result

@benchmark_core.register_test
def test_repeat():
    generate_pt_test(
        [short_config],
        map_dims_to_shape,
        [('repeat', torch_repeat_dummy)]
    )

if __name__ == "__main__":
    total_time_per_iter_s = benchmark_runner.main()
    achieved_bandwidth = (total_bytes * BYTES_TO_MB) / total_time_per_iter_s
    print("Achieved Bandwidth:{} MB/s".format(achieved_bandwidth))
