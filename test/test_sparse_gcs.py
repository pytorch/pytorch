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
import math
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
        sp = torch.tensor([[1, 2], [3, 4]]).to_sparse_gcs(None, -999)
        self.assertEqual(torch.tensor([0, 2, 4], dtype=torch.int32), sp.pointers())
        self.assertEqual(torch.tensor([0, 1, 0, 1], dtype=torch.int32), sp.indices())
        self.assertEqual(torch.tensor([1, 2, 3, 4], dtype=torch.int64), sp.values())

    def test_dense_convert(self):
        size = (5, 5, 5, 5, 5)
        dense = torch.randn(size)
        sparse = dense.to_sparse_gcs(None, -999)
        self.assertEqual(sparse.to_dense(), dense)
        
        size = (4, 6)
        dense = torch.randn(size)
        sparse = dense.to_sparse_gcs(None, -999)
        self.assertEqual(sparse.to_dense(), dense)

        size = (3, 2, 2, 2, 6)
        dense = torch.randn(size)
        sparse = dense.to_sparse_gcs(None, -999)
        self.assertEqual(sparse.to_dense(), dense)

    def test_gcs_matvec(self):
        side = 100
        gcs = self.gen_sparse_gcs((side, side), 1000)
        vec = torch.randn(side, dtype=torch.double)

        res = gcs.matmul(vec)
        expected = gcs.to_dense().matmul(vec)

        self.assertEqual(res, expected)
        
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

if __name__ == '__main__':
    run_tests()
