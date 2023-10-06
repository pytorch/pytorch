import functools
import operator
import random
import time

import numpy as np
import torch


# shim for torch.cuda.Event when running on cpu
class Event:
    def __init__(self, enable_timing):
        pass

    def record(self):
        self.time = time.perf_counter()

    def elapsed_time(self, end_event):
        assert isinstance(end_event, Event)
        return end_event.time - self.time


def gen_sparse_csr(shape, nnz):
    fill_value = 0
    total_values = functools.reduce(operator.mul, shape, 1)
    dense = np.random.randn(total_values)
    fills = random.sample(list(range(total_values)), total_values - nnz)

    for f in fills:
        dense[f] = fill_value
    dense = torch.from_numpy(dense.reshape(shape))

    return dense.to_sparse_csr()


def gen_sparse_coo(shape, nnz):
    dense = np.random.randn(*shape)
    values = []
    indices = [[], []]
    for n in range(nnz):
        row = random.randint(0, shape[0] - 1)
        col = random.randint(0, shape[1] - 1)
        indices[0].append(row)
        indices[1].append(col)
        values.append(dense[row, col])

    return torch.sparse_coo_tensor(indices, values, size=shape)


def gen_sparse_coo_and_csr(shape, nnz):
    total_values = functools.reduce(operator.mul, shape, 1)
    dense = np.random.randn(total_values)
    fills = random.sample(list(range(total_values)), total_values - nnz)

    for f in fills:
        dense[f] = 0

    dense = torch.from_numpy(dense.reshape(shape))
    return dense.to_sparse(), dense.to_sparse_csr()
