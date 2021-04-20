import torch
import functools
import random
import operator
import warnings
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyCPU)

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestSparseCSR(TestCase):
    def gen_sparse_csr(self, shape, nnz, dtype, device, index_dtype=torch.int64):
        total_values = functools.reduce(operator.mul, shape, 1)
        dense = torch.randn(total_values, dtype=dtype, device=device)
        fills = random.sample(list(range(total_values)), total_values - nnz)

        dense[fills] = 0
        dense = dense.reshape(shape)
        s = dense.to_sparse_csr()
        return torch.sparse_csr_tensor(torch.tensor(s.crow_indices(), dtype=index_dtype),
                                       torch.tensor(s.col_indices(), dtype=index_dtype),
                                       torch.tensor(s.values()), size=shape, dtype=dtype, device=device)

    def setUp(self):
        self.index_tensor = lambda *args: torch.tensor(*args, dtype=torch.int32)

    @onlyCPU
    def test_csr_layout(self, device):
        self.assertEqual(str(torch.sparse_csr), 'torch.sparse_csr')
        self.assertEqual(type(torch.sparse_csr), torch.layout)

    @onlyCPU
    @dtypes(torch.double)
    def test_sparse_csr_constructor_shape_inference(self, device, dtype):
        crow_indices = [0, 2, 4]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]
        sparse = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int64),
                                         torch.tensor(col_indices, dtype=torch.int64),
                                         torch.tensor(values), dtype=dtype, device=device)
        self.assertEqual(torch.tensor(crow_indices, dtype=torch.int64), sparse.crow_indices())
        self.assertEqual((len(crow_indices) - 1, max(col_indices) + 1), sparse.shape)

    @onlyCPU
    @dtypes(torch.float)
    def test_sparse_csr_constructor(self, device, dtype):
        crow_indices = [0, 2, 4]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]

        sparse = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=torch.int32),
                                         torch.tensor(col_indices, dtype=torch.int32),
                                         torch.tensor(values), size=(2, 10), dtype=dtype, device=device)

        self.assertEqual((2, 10), sparse.shape)
        self.assertEqual(torch.tensor(crow_indices, dtype=torch.int32), sparse.crow_indices())

    @onlyCPU
    @dtypes(torch.double)
    def test_sparse_csr_print(self, device, dtype):
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

            x = self.gen_sparse_csr(shape, nnz, dtype, device)

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

    @onlyCPU
    def test_sparse_csr_from_dense(self, device):
        sp = torch.tensor([[1, 2], [3, 4]], device=device).to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 4], dtype=torch.int64), sp.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0, 1], dtype=torch.int64), sp.col_indices())
        self.assertEqual(torch.tensor([1, 2, 3, 4], dtype=torch.int64), sp.values())

        dense = torch.tensor([[4, 5, 0], [0, 0, 0], [1, 0, 0]], device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 2, 3], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([4, 5, 1]), sparse.values())

        dense = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]], device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 0, 1, 2], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([2, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([1, 1]), sparse.values())

        dense = torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 3, 6, 9], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 2] * 3, dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([2] * 9), sparse.values())

    @onlyCPU
    @dtypes(torch.double)
    def test_dense_convert(self, device, dtype):
        size = (5, 5)
        dense = torch.randn(size, dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(sparse.to_dense(), dense)

        size = (4, 6)
        dense = torch.randn(size, dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(sparse.to_dense(), dense)

        crow_indices = torch.tensor([0, 3, 5])
        col_indices = torch.tensor([0, 1, 2, 0, 1])
        values = torch.tensor([1, 2, 1, 3, 4], dtype=dtype)
        csr = torch.sparse_csr_tensor(crow_indices, col_indices,
                                      values, dtype=dtype, device=device)
        dense = torch.tensor([[1, 2, 1], [3, 4, 0]], dtype=dtype, device=device)
        self.assertEqual(csr.to_dense(), dense)

    @onlyCPU
    @dtypes(torch.double)
    def test_coo_to_csr_convert(self, device, dtype):
        size = (5, 5)
        dense = torch.randn(size, dtype=dtype, device=device)
        sparse_coo = dense.to_sparse()
        sparse_csr = sparse_coo.to_sparse_csr()

        self.assertTrue(sparse_csr.is_sparse_csr)
        self.assertEqual(sparse_csr.to_dense(), dense)

        vec = torch.randn((5, 1), dtype=dtype, device=device)
        coo_product = sparse_coo.matmul(vec)
        csr_product = sparse_csr.matmul(vec)

        self.assertEqual(coo_product, csr_product)

        vec = torch.randn((100, 1), dtype=dtype, device=device)
        index = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        coo = torch.sparse_coo_tensor(index, values, torch.Size([100, 100]), dtype=dtype, device=device)
        csr = coo.to_sparse_csr()

        self.assertEqual(coo.matmul(vec), csr.matmul(vec))

    @onlyCPU
    @dtypes(torch.double)
    def test_mkl_matvec_warnings(self, device, dtype):
        if torch.has_mkl:
            sp = torch.sparse_csr_tensor(torch.tensor([0, 2, 4]),
                                         torch.tensor([0, 1, 0, 1]),
                                         torch.tensor([1, 2, 3, 4], dtype=dtype, device=device))
            vec = torch.randn((2, 1), dtype=dtype, device=device)
            with warnings.catch_warnings(record=True) as w:
                sp.matmul(vec)
                self.assertEqual(len(w), 2)

    @dtypes(torch.double)
    def test_dense_convert_error(self, device, dtype):
        size = (4, 2, 4)
        dense = torch.randn(size, dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, "Only 2D"):
            sparse = dense.to_sparse_csr()

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_csr_matvec(self, device, dtype):
        side = 100
        for index_dtype in [torch.int32, torch.int64]:
            csr = self.gen_sparse_csr((side, side), 1000, dtype, device, index_dtype=index_dtype)
            vec = torch.randn(side, dtype=dtype, device=device)

            res = csr.matmul(vec)
            expected = csr.to_dense().matmul(vec)

            self.assertEqual(res, expected)

            bad_vec = torch.randn(side + 10, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, "mv: expected"):
                csr.matmul(bad_vec)

    @onlyCPU
    @dtypes(*torch.testing.floating_and_complex_types())
    def test_coo_csr_conversion(self, device, dtype):
        size = (5, 5)
        dense = torch.randn(size, dtype=dtype, device=device)
        coo_sparse = dense.to_sparse()
        csr_sparse = coo_sparse.to_sparse_csr()

        self.assertEqual(csr_sparse.to_dense(), dense)


# e.g., TestSparseCSRCPU and TestSparseCSRCUDA
instantiate_device_type_tests(TestSparseCSR, globals())

if __name__ == '__main__':
    run_tests()
