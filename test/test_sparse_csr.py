import torch

# NOTE: These tests are inspired from test_sparse.py and may duplicate some behaviour.
# Need to think about merging them both sometime down the line.

# Major differences between testing of CSR and COO is that we don't need to test CSR
# for coalesced/uncoalesced behaviour.

# TODO: remove this global setting
# Sparse tests use double as the default dtype
torch.set_default_dtype(torch.double)

import itertools
import functools
import random
import unittest
import operator
import numpy as np
from collections import defaultdict
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestSparseGCS(TestCase):    
    def make_sparse_gcs(self, data, reduction=None, fill_value=float('NaN')):
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
        strides2 = make_strides(shape[l-1:-1])
        # print(f'{shape} {strides1} {strides2} {dims1} {dims2}')
        # <row>: <list of (colindex, value)>
        col_value = defaultdict(list)
        for index in itertools.product(*map(range, shape)):
            v = data
            for i in index:
                v = v[i]
            if v == fill_value or np.isnan(v):
                continue
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
            c, v = zip(*cv)
            co.extend(c)
            values.extend(v)

        # print(f"{torch.tensor(ro)} {torch.tensor(co)} {torch.tensor(values)} {torch.tensor(reduction)} {shape}")
        return torch.sparse_gcs_tensor(torch.tensor(ro), torch.tensor(co), torch.tensor(values),
                                       torch.tensor(reduction), shape, fill_value)

    def gen_sparse_gcs(self, shape, nnz, fill_value=float('NaN')):
        total_values = functools.reduce(operator.mul, shape, 1)
        dense = np.random.randn(total_values)
        fills = random.sample(list(range(total_values)), total_values-nnz)

        for f in fills:
            dense[f] = fill_value
        dense = dense.reshape(shape)

        return self.make_sparse_gcs(dense, None, fill_value)
    
    def setUp(self):
        # These parameters control the various ways we can run the test.
        # We will subclass and override this method to implement CUDA
        # tests
        self.is_cuda = False
        self.is_uncoalesced = False
        self.device = 'cpu'
        self.exact_dtype = True
    
    def test_gcs_layout(self):
        self.assertEqual(str(torch.sparse_gcs), 'torch.sparse_gcs')
        self.assertEqual(type(torch.sparse_gcs), torch.layout)

    def test_sparse_gcs_from_dense(self):
        sp = self.make_sparse_gcs([[1, 2], [3, 4]])

        self.assertEqual(torch.tensor([0, 2, 4]), sp.pointers())
        self.assertEqual(torch.tensor([0, 1, 0, 1]), sp.indices())
        self.assertEqual(torch.tensor([1, 2, 3, 4]), sp.values())

    def test_dense_convert(self):
        rand = np.random.randn(100, 100)
        sparse = self.make_sparse_gcs(rand)

        self.assertEqual(sparse.to_dense(), rand)

        multi_dim = np.random.randn(10, 8, 50, 20, 300)
        sparse = self.make_sparse_gcs(multi_dim)

        self.assertEqual(sparse.to_dense(), multi_dim)

    def test_basic_ops(self):
        # sparse-sparse addition
        x1 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 100)
        x2 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 140)
        y1 = x1 + x2
        expected = x1.to_dense() * x2.to_dense()
        self.assertEqual(y1.to_dense(), expected)

        # sparse-dense addition
        x1 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 100)
        x2 = torch.ones((10, 3, 40, 5, 2))
        y1 = x1 + x2
        
        expected = x1.to_dense() + x2.to_dense()
        self.assertEqual(y1.to_dense(), expected)

        # sparse-scalar addition
        x1 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 100)
        y1 = x1 + 18.3
        expected = x1.to_dense() + 18.3
        self.assertEqual(y1.to_dense(), expected)
        
        # sparse-sparse multiplication
        x1 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 100)
        x2 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 140)
        y1 = x1 * x2
        expected = x1.to_dense() * x2.to_dense()

        self.assertEqual(y1.to_dense(), expected)

        # sparse-scalar multiplication
        x1 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 100)
        y1 = x1 * 18.3
        expected = x1.to_dense() * 18.3

        self.assertEqual(y1.to_dense(), expected)
        
    def test_autograd(self):
        pass
    
    def test_sparse_gcs_constructor(self):
        pass

if __name__ == '__main__':
    run_tests()
