import torch

torch.set_default_dtype(torch.double)

import functools
import random
import operator
import numpy as np
import warnings
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestSparseCSR(TestCase):
    def gen_sparse_csr(self, shape, nnz):
        total_values = functools.reduce(operator.mul, shape, 1)
        dense = np.random.randn(total_values)
        fills = random.sample(list(range(total_values)), total_values - nnz)

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
        self.index_tensor = lambda *args: torch.tensor(*args, dtype=torch.int32)
        self.value_tensor = lambda *args: torch.tensor(*args, dtype=torch.double)

    def test_csr_layout(self):
        self.assertEqual(str(torch.sparse_csr), 'torch.sparse_csr')
        self.assertEqual(type(torch.sparse_csr), torch.layout)

    def test_sparse_csr_constructor_shape_inference(self):
        crow_indices = [0, 2, 4]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]
        sparse = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
                                         torch.tensor(col_indices, dtype=torch.int64),
                                         torch.tensor(values), dtype=torch.double)
        self.assertEqual(torch.tensor(crow_indices, dtype=torch.int64), sparse.crow_indices())
        self.assertEqual((len(crow_indices) - 1, max(col_indices) + 1), sparse.shape)

    def test_sparse_csr_constructor(self):
        crow_indices = [0, 2, 4]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]

        sparse = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int32),
                                         torch.tensor(col_indices, dtype=torch.int32),
                                         torch.tensor(values), size=(2, 10), dtype=torch.float)

        self.assertEqual((2, 10), sparse.shape)
        self.assertEqual(torch.tensor(crow_indices, dtype=torch.int32), sparse.crow_indices())

    def test_sparse_csr_print(self):
        shape_nnz = [
            ((1000, 10), 10)
        ]

        printed = []
        for shape, nnz in shape_nnz:
            values_shape = torch.Size((nnz,))
            col_indices_shape = torch.Size((nnz,))
            crow_indices_shape = torch.Size((shape[0] + 1,))
            printed.append("# shape: {}".format(torch.Size(shape)))
            printed.append("# nnz: {}".format(nnz))
            printed.append("# crow_indices shape: {}".format(crow_indices_shape))
            printed.append("# col_indices shape: {}".format(col_indices_shape))
            printed.append("# values_shape: {}".format(values_shape))

            x = self.gen_sparse_csr(shape, nnz)

            printed.append("# sparse tensor")
            printed.append(str(x))
            printed.append("# _crow_indices")
            printed.append(str(x.crow_indices()))
            printed.append("# _col_indices")
            printed.append(str(x.col_indices()))
            printed.append("# _values")
            printed.append(str(x.values()))
            printed.append('')

            self.assertEqual(len(printed) > 0, True)

    def test_sparse_csr_from_dense(self):
        sp = torch.tensor([[1, 2], [3, 4]]).to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 4], dtype=torch.int64), sp.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0, 1], dtype=torch.int64), sp.col_indices())
        self.assertEqual(torch.tensor([1, 2, 3, 4], dtype=torch.int64), sp.values())

        dense = torch.tensor([[4, 5, 0], [0, 0, 0], [1, 0, 0]])
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 2, 3], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([4, 5, 1]), sparse.values())

        dense = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 0, 1, 2], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([2, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([1, 1]), sparse.values())

        dense = torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 3, 6, 9], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 2] * 3, dtype=torch.int64), sparse.col_indices())
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

        crow_indices = torch.tensor([0, 3, 5])
        col_indices = torch.tensor([0, 1, 2, 0, 1])
        values = torch.tensor([1, 2, 1, 3, 4])
        csr = torch.sparse_csr_tensor(crow_indices, col_indices,
                                      values, dtype=torch.double)
        dense = torch.tensor([[1, 2, 1], [3, 4, 0]], dtype=torch.double)
        self.assertEqual(csr.to_dense(), dense)

    def test_coo_to_csr_convert(self):
        size = (5, 5)
        dense = torch.randn(size)
        sparse_coo = dense.to_sparse()
        sparse_csr = sparse_coo.to_sparse_csr()

        self.assertTrue(sparse_csr.is_sparse_csr)
        self.assertEqual(sparse_csr.to_dense(), dense)

        vec = torch.randn((5, 1))
        coo_product = sparse_coo.matmul(vec)
        csr_product = sparse_csr.matmul(vec)

        self.assertEqual(coo_product, csr_product)

        vec = torch.randn((100, 1))
        index = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        values = self.value_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        coo = torch.sparse_coo_tensor(index, values, torch.Size([100, 100]))
        csr = coo.to_sparse_csr()

        self.assertEqual(coo.matmul(vec), csr.matmul(vec))

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
