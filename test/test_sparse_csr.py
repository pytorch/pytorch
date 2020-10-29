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
    def gen_sparse_gcs(self, shape, nnz, fill_value=float('NaN')):
        total_values = functools.reduce(operator.mul, shape, 1)
        dense = np.random.randn(total_values)
        fills = random.sample(list(range(total_values)), total_values-nnz)

        for f in fills:
            dense[f] = fill_value
        dense = torch.from_numpy(dense.reshape(shape))
        print(dense.shape)

        return dense.to_sparse_gcs(None, fill_value)
    
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
        sp = torch.tensor([[1, 2], [3, 4]]).to_sparse_gcs(None, None)

        self.assertEqual(torch.tensor([0, 2, 4]), sp.pointers())
        self.assertEqual(torch.tensor([0, 1, 0, 1]), sp.indices())
        self.assertEqual(torch.tensor([1, 2, 3, 4]), sp.values())

    def test_dense_convert(self):
        ones = np.ones((5, 5, 5, 5, 5))
        sparse = self.make_sparse_gcs(ones)
        self.assertEqual(sparse.to_dense(), ones)
        
        rand = np.random.randn(4, 6)
        sparse = self.make_sparse_gcs(rand)
        self.assertEqual(sparse.to_dense(), rand)

        multi_dim = np.random.randn(3, 2, 2, 2, 6)
        sparse = self.make_sparse_gcs(multi_dim)
        self.assertEqual(sparse.to_dense(), multi_dim)

    def test_gcs_matvec(self):
        side = 1000
        gcs = self.gen_sparse_gcs((side, side), 1000)
        vec = torch.randn(side, dtype=torch.double)

        res = gcs.matmul(vec)
        expected = gcs.to_dense().matmul(vec)

        self.assertEqual(res, expected)

    def test_gcs_segfault(self):
        side1 = 100
        side2 = 120
        nnz = 100
        k = 100
        
        gcs = self.gen_sparse_gcs((side1, k), nnz)
        mat = torch.randn((k, side2), dtype=torch.double)
        res = gcs.matmul(mat)
        
        gcs.to_dense().matmul(mat)
        
    def test_gcs_matmul(self):
        side1 = 100
        side2 = 120
        nnz = 100
        for k in [5, 50, 100, 200]:
            gcs = self.gen_sparse_gcs((side1, k), nnz)
            mat = torch.randn((k, side2), dtype=torch.double)

            res = gcs.matmul(mat)
            expected = gcs.to_dense().matmul(mat)

            self.assertEqual(res, expected)

    def test_basic_elementwise_ops(self):
        # sparse-sparse addition
        x1 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 100)
        x2 = self.gen_sparse_gcs((10, 3, 40, 5, 2), 140)
        y1 = x1 + x2
        expected = x1.to_dense() + x2.to_dense()
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

if __name__ == '__main__':
    run_tests()
