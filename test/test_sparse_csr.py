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
import warnings
from collections import defaultdict
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestSparseCSR(TestCase):
    def gen_sparse_csr(self, shape, nnz):
        total_values = functools.reduce(operator.mul, shape, 1)
        dense = np.random.randn(total_values)
        fills = random.sample(list(range(total_values)), total_values-nnz)

        for f in fills:
            dense[f] = 0
        dense = torch.from_numpy(dense.reshape(shape))

        return dense.to_sparse_csr()
    
    def setUp(self):
        # These parameters control the various ways we can run the test.
        # We will subclass and override this method to implement CUDA
        # tests
        self.is_cuda = False
        self.device = 'cpu'
    
    def test_csr_layout(self):
        self.assertEqual(str(torch.sparse_csr), 'torch.sparse_csr')
        self.assertEqual(type(torch.sparse_csr), torch.layout)

    def test_sparse_csr_constructor_shape_inference(self):
        crow_indices = [0, 3, 6, 9]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]
        sparse = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64), 
                                         torch.tensor(col_indices, dtype=torch.int64), 
                                         torch.tensor(values), dtype=torch.double)

        print(sparse.shape)
        self.assertEqual(torch.tensor(crow_indices, dtype=torch.int64), sparse.crow_indices())
        self.assertEqual((len(crow_indices) - 1, max(col_indices) + 1), sparse.shape)

    def test_sparse_csr_constructor(self):
        crow_indices = [0, 3, 6, 9]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]

        sparse = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int32),
                                         torch.tensor(col_indices, dtype=torch.int32),
                                         torch.tensor(values), size=(3, 10), dtype=torch.float)

        self.assertEqual((3, 10), sparse.shape)
        self.assertEqual(torch.tensor(crow_indices, dtype=torch.int32), sparse.crow_indices())

    def test_sparse_csr_print(self):
        shape_nnz = [
            ((1000, 10), 10)
        ]

        printed = []
        for shape, nnz in shape_nnz:
            values_shape = torch.Size((nnz,))
            col_indices_shape = torch.Size((nnz,))
            crow_indices_shape = torch.Size((shape[0]+1,))
            

    def test_sparse_csr_from_dense(self):
        sp = torch.tensor([[1, 2], [3, 4]]).to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 4], dtype=torch.int32), sp.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0, 1], dtype=torch.int32), sp.col_indices())
        self.assertEqual(torch.tensor([1, 2, 3, 4], dtype=torch.int64), sp.values())

        dense = torch.tensor([[4, 5, 0], [0, 0, 0], [1, 0, 0]])
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 2, 3], dtype=torch.int32), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0], dtype=torch.int32), sparse.col_indices())
        self.assertEqual(torch.tensor([4, 5, 1]), sparse.values())

        dense = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 0, 1, 2], dtype=torch.int32), sparse.crow_indices())
        self.assertEqual(torch.tensor([2, 0], dtype=torch.int32), sparse.col_indices())
        self.assertEqual(torch.tensor([1, 1]), sparse.values())

        dense = torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 3, 6, 9], dtype=torch.int32), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 2] * 3, dtype=torch.int32), sparse.col_indices())
        self.assertEqual(torch.tensor([2] * 9), sparse.values())

    def test_dense_convert(self):
        size = (5, 5)
        dense = torch.randn(size)
        sparse = dense.to_sparse_csr()
        self.assertEqual(sparse.to_dense(), dense)
        
        size = (4, 6)
        dense = torch.randn(size)
        sparse = dense.to_sparse_csr()
        self.assertEqual(sparse.to_dense(), dense)

    def test_mkl_matvec_warnings(self):
        if torch.has_mkl:
            sp = torch.sparse_csr_tensor(torch.tensor([0, 2, 4]),
                                        torch.tensor([0, 1, 0, 1]),
                                        torch.tensor([1, 2, 3, 4], dtype=torch.double))
            vec = torch.randn((2, 1))
            with warnings.catch_warnings(record=True) as w:
                sp.matmul(vec)
                self.assertEqual(len(w), 2)

    def test_dense_convert_error(self):
        size = (4, 2, 4)
        dense = torch.randn(size)

        with self.assertRaisesRegex(RuntimeError, "Only 2D"):
            sparse = dense.to_sparse_csr()

    def test_csr_matvec(self):
        side = 100
        csr = self.gen_sparse_csr((side, side), 1000)
        vec = torch.randn(side, dtype=torch.double)

        res = csr.matmul(vec)
        expected = csr.to_dense().matmul(vec)

        self.assertEqual(res, expected)

        bad_vec = torch.randn(side + 10, dtype=torch.double)
        with self.assertRaisesRegex(RuntimeError, "mv: expected"):
            csr.matmul(bad_vec)

    def test_coo_csr_conversion(self):
        size = (5, 5)
        dense = torch.randn(size)
        coo_sparse = dense.to_sparse()
        csr_sparse = coo_sparse.to_sparse_csr()

        self.assertEqual(csr_sparse.to_dense(), dense)

if __name__ == '__main__':
    run_tests()
