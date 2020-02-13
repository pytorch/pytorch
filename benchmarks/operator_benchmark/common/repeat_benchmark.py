from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch

import time

"""Microbenchmarks for Tensor repeat operator. Supports PyTorch."""

input_shapes = (
               (4, 4, 1),
               (16, 1, 32),
               (64, 64, 1, 1),
               (8, 256, 128),
               (1, 64, 128, 32),
               (512, 512),
)

repeats = (
          (1, 1, 1, 64),
          (1, 4, 1, 2),
          (1, 2, 2, 15),
          (1, 1, 3, 2),
          (128, 1, 8, 1),
          (1, 1, 2, 16),
)

NUM_WARMUP_ITERS = 5
NUM_BENCHMARK_ITERS = 10
DTYPE_TO_BYTES = {'float' : 4}

def generate_data_for_repeat():
    input_tensors = [torch.randn(*input_shape) for input_shape in input_shapes]
    total_num_elements = 0
    for input_tensor, repeat in zip(input_tensors, repeats):
        total_num_elements += input_tensor.numel()
        total_num_elements += input_tensor.numel() * np.prod(repeat)
    return input_tensors, (total_num_elements * DTYPE_TO_BYTES['float'])

input_tensors, total_bytes = generate_data_for_repeat()
BYTES_TO_MB = (1. / 1000. / 1000.)

def pt_repeat(input_tensor, repeat):
    return input_tensor.repeat(repeat)

def pt_repeat_n_times(niters):
    for _ in range(niters):
        for input_tensor, repeat in zip(input_tensors, repeats):
            pt_repeat(input_tensor, repeat)

if __name__ == "__main__":
    # Warm up runs.
    pt_repeat_n_times(NUM_WARMUP_ITERS)
    s = time.time()
    pt_repeat_n_times(NUM_BENCHMARK_ITERS)
    total_time_s = (time.time() - s)
    total_time_per_iter_s = total_time_s / NUM_BENCHMARK_ITERS
    achieved_bandwidth = (total_bytes * BYTES_TO_MB) / total_time_per_iter_s
    print("Time:{} Achieved Bandwidth:{} MB/s".format(total_time_per_iter_s, achieved_bandwidth))
