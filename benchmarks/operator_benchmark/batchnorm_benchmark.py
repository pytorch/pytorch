from __future__ import absolute_import, division, print_function, unicode_literals

import time

import numpy
import torch
import torch.nn.functional as F


def benchmark_batch_norm(data_shape):
    C = data_shape[1]
    x = torch.rand(data_shape)
    mean = torch.rand(C)
    var = torch.rand(C)
    weight = torch.rand(C)
    bias = torch.rand(C)
    NITER = 10000
    input_size = numpy.prod(data_shape)
    total_size = 2 * input_size + 4 * C
    for i in range(-10, NITER):
        if i == 0:
            s = time.time()
        F.batch_norm(x, mean, var, weight, bias)
    elapsed_sec = (time.time() - s) / NITER
    print(
        "batch_norm: data shape: %s, bandwidth: %.2f GB/s"
        % (data_shape, (total_size * 4) / elapsed_sec / 1e9)
    )


def main():
    data_shapes = [[1, 256, 3136], [1, 2 ** 16, 1], [128, 2048, 1]]
    for data_shape in data_shapes:
        benchmark_batch_norm(data_shape)


if __name__ == "__main__":
    main()
