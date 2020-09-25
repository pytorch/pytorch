import torch
import itertools
import functools
import random
import unittest
import operator
import numpy as np
from collections import defaultdict
import time

# shim for torch.cuda.Event when running on cpu
class Event(object):
    def __init__(self, enable_timing):
        pass

    def record(self):
        self.time = time.perf_counter()

    def elapsed_time(self, end_event):
        assert isinstance(end_event, Event)
        return end_event.time - self.time

def make_sparse_gcs(data, reduction=None, fill_value=float('NaN')):
    def get_shape(data):
        if isinstance(data, (list, tuple)):
            dims = len(data)
            if dims == 0:
                return (0,)
            return (dims, ) + get_shape(data[0])
        elif isinstance(data, np.ndarray):
            return data.shape
        return ()

    def make_strides(shape, dims=None):
        if dims is None:
            dims = tuple(range(len(shape)))
            ndims = len(dims)
        if ndims == 0:
            return ()
        strides = [1]
        for i in range(ndims - 1):
            strides.insert(0, strides[0] * shape[dims[ndims - i - 1]])
        return tuple(strides)

    def apply_reduction(index, strides, dims):
        return sum(strides[k] * index[dims[k]] for k in range(len(dims)))

    shape = get_shape(data)
    N = len(shape)
    # TODO: N=0, N=1
    if reduction is None:
        dims1 = tuple(range(N//2))
        dims2 = tuple(range(N//2, N))
        reduction = dims1 + dims2 + (N//2,)
        l = N // 2
    else:
        l = reduction[-1]
        dims1 = reduction[:l]
        dims2 = reduction[l:-1]

    strides1 = make_strides(shape[:l])
    strides2 = make_strides(shape[l:])
    # <row>: <list of (colindex, value)>
    col_value = defaultdict(list)
    for index in itertools.product(*map(range, shape)):
        v = data
        for i in index:
            v = v[i]
        if v == fill_value or np.isnan(v):
            continue
        # print(index)
        p1 = apply_reduction(index, strides1, dims1)
        p2 = apply_reduction(index, strides2, dims2)
        col_value[p1].append((p2, v))
        ro = [0]
        co = []
        values = []

    for i in range(max(col_value)+1):
        cv = col_value.get(i, [])
        ro.append(ro[-1] + len(cv))
        cv.sort()
        if len(cv) != 0:
            c, v = zip(*cv)
            co.extend(c)
            values.extend(v)

    return torch.sparse_gcs_tensor(torch.tensor(ro, dtype=torch.int32),
                                   torch.tensor(co, dtype=torch.int32), torch.tensor(values),
                                   torch.tensor(reduction), shape, fill_value)

def gen_sparse_gcs(shape, nnz, fill_value=float('NaN')):
    total_values = functools.reduce(operator.mul, shape, 1)
    dense = np.random.randn(total_values)
    fills = random.sample(list(range(total_values)), total_values-nnz)

    for f in fills:
        dense[f] = fill_value
    dense = dense.reshape(shape)

    return make_sparse_gcs(dense, None, fill_value)

def gen_sparse_coo(shape, nnz):
    dense = np.random.randn(*shape)
    values = []
    indices = [[], []]
    for n in range(nnz):
        row = random.randint(0, shape[0]-1)
        col = random.randint(0, shape[1]-1)
        indices[0].append(row)
        indices[1].append(col)
        values.append(dense[row, col])

    return torch.sparse_coo_tensor(indices, values, [shape[0], shape[1]])
