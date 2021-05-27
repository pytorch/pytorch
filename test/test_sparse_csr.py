import torch
import warnings
import unittest
import random
import itertools
from torch.testing._internal.common_utils import \
    (IS_MACOS, IS_WINDOWS, TestCase, run_tests, load_tests, coalescedonoff, make_tensor)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyCPU, onlyCUDA)

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestSparseCSR(TestCase):

    @onlyCPU
    def test_csr_layout(self, _):
        self.assertEqual(str(torch.sparse_csr), 'torch.sparse_csr')
        self.assertEqual(type(torch.sparse_csr), torch.layout)

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
        self.assertEqual(dtype, sparse.dtype)
        self.assertEqual(torch.device(device), sparse.device)

    @dtypes(*torch.testing.get_all_dtypes(include_bool=False, include_half=False,
                                          include_bfloat16=False, include_complex=False))
    def test_sparse_csr_constructor(self, device, dtype):
        crow_indices = [0, 2, 4]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]
        for index_dtype in [torch.int32, torch.int64]:
            sparse = torch.sparse_csr_tensor(torch.tensor(crow_indices, dtype=index_dtype),
                                             torch.tensor(col_indices, dtype=index_dtype),
                                             torch.tensor(values),
                                             size=(2, 10),
                                             dtype=dtype,
                                             device=device)
            self.assertEqual((2, 10), sparse.shape)
            self.assertEqual(torch.tensor(crow_indices, dtype=index_dtype), sparse.crow_indices())
            self.assertEqual(torch.tensor(col_indices, dtype=index_dtype), sparse.col_indices())
            self.assertEqual(torch.tensor(values, dtype=dtype), sparse.values())

    @dtypes(*torch.testing.get_all_dtypes(include_bool=False, include_half=False,
                                          include_bfloat16=False, include_complex=False))
    def test_sparse_csr_constructor_from_lists(self, device, dtype):
        # without size
        sparse = torch.sparse_csr_tensor([0, 2, 4],
                                         [0, 1, 0, 1],
                                         [1, 2, 3, 4],
                                         dtype=dtype,
                                         device=device)

        self.assertEqual((2, 2), sparse.shape)
        self.assertEqual(4, sparse.numel())
        self.assertEqual(torch.tensor([0, 2, 4], dtype=torch.int64, device=device), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=device), sparse.col_indices())
        self.assertEqual(torch.tensor([1, 2, 3, 4], dtype=dtype, device=device), sparse.values())

        # with size
        for sparse_csr_tensor in [torch.sparse_csr_tensor, torch._sparse_csr_tensor_unsafe]:
            sparse = sparse_csr_tensor([0, 2, 4],
                                       [0, 1, 0, 1],
                                       [1, 2, 3, 4],
                                       size=(2, 10),
                                       dtype=dtype,
                                       device=device)

            self.assertEqual((2, 10), sparse.shape)
            self.assertEqual(torch.tensor([0, 2, 4], dtype=torch.int64, device=device), sparse.crow_indices())
            self.assertEqual(torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=device), sparse.col_indices())
            self.assertEqual(torch.tensor([1, 2, 3, 4], dtype=dtype, device=device), sparse.values())

    def test_factory_type_invariants_check(self, device):
        with self.assertRaisesRegex(RuntimeError, "both crow_indices and col_indices should have the same type."):
            torch.sparse_csr_tensor(torch.tensor([0, 2, 4], dtype=torch.int64),
                                    torch.tensor([0, 1, 0, 1], dtype=torch.int32),
                                    torch.tensor([1, 2, 3, 4]),
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"\"csr_construct_check\" not implemented for 'Short'"):
            torch.sparse_csr_tensor(torch.tensor([0, 2, 4], dtype=torch.int16),
                                    torch.tensor([0, 1, 0, 1], dtype=torch.int16),
                                    torch.tensor([1, 2, 3, 4]),
                                    device=device)

    def test_factory_layout_invariants_check(self, device):
        with self.assertRaisesRegex(RuntimeError, "expected values to be a strided and contiguous tensor"):
            values = torch.tensor([1.], device=device).expand(4,)
            torch.sparse_csr_tensor(torch.tensor([0, 2, 4], device=device),
                                    torch.tensor([0, 1, 0, 1], device=device),
                                    values)

        with self.assertRaisesRegex(RuntimeError, "expected col_indices to be a strided and contiguous tensor"):
            col_indices = torch.tensor([0], device=device).expand(4,)
            torch.sparse_csr_tensor(torch.tensor([0, 2, 4]),
                                    col_indices,
                                    torch.tensor([1, 2, 3, 4]))

        with self.assertRaisesRegex(RuntimeError, "expected crow_indices to be a strided and contiguous tensor"):
            crow_indices = torch.arange(6, device=device)
            torch.sparse_csr_tensor(crow_indices[::2],
                                    torch.tensor([0, 1, 0, 1], device=device),
                                    torch.tensor([1, 2, 3, 4]))

    def test_factory_shape_invariants_check(self, device):
        crow_indices = [0, 2, 4]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]
        size = (2, 10)
        torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor(col_indices), torch.tensor(values), size,
                                device=device)

        with self.assertRaisesRegex(RuntimeError, r"size of a CSR tensor must be of length 2, but got: 3"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor(col_indices), torch.tensor(values),
                                    size=(2, 10, 2),
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"crow_indices must have dim\=1 but got crow_indices\.dim\(\)\=2"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices).repeat(2, 1),
                                    torch.tensor(col_indices),
                                    torch.tensor(values),
                                    size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"col_indices must have dim\=1 but got col_indices\.dim\(\)\=2"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices),
                                    torch.tensor(col_indices).repeat(2, 1),
                                    torch.tensor(values),
                                    size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"values must have dim\=1 but got values\.dim\(\)\=2"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices),
                                    torch.tensor(col_indices),
                                    torch.tensor(values).repeat(2, 1),
                                    size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    r"crow_indices\.numel\(\) must be size\(0\) \+ 1, but got: 3"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor(col_indices), torch.tensor(values), (1, 1),
                                    device=device)


        with self.assertRaisesRegex(RuntimeError,
                                    r"col_indices and values must have equal sizes, " +
                                    r"but got col_indices\.numel\(\): 3, values\.numel\(\): 4"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor([0, 1, 0]), torch.tensor(values), size,
                                    device=device)

    def test_factory_indices_invariants_check(self, device):
        crow_indices = [0, 2, 4]
        col_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]
        size = (2, 10)
        with self.assertRaisesRegex(RuntimeError, "0th value of crow_indices must be 0."):
            torch.sparse_csr_tensor(torch.tensor([-1, 0, 4]), torch.tensor(col_indices), torch.tensor(values), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    "last value of crow_indices should be equal to the length of col_indices."):
            torch.sparse_csr_tensor(torch.tensor([0, 2, 5]), torch.tensor(col_indices), torch.tensor(values), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    r"at position i \= 2," +
                                    r" this condition crow_indices\[i - 1\] <\= crow_indices\[i\] fails"):
            torch.sparse_csr_tensor(torch.tensor([0, 5, 4]), torch.tensor(col_indices), torch.tensor(values), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"col_indices\.min\(\) should be greater or equal to zero"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor([0, -1, 0, 1]), torch.tensor(values), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"size\(1\) should be greater than col_indices\.max\(\)"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor([0, 11, 0, 1]), torch.tensor(values), size,
                                    device=device)

    @onlyCUDA
    @dtypes(*torch.testing.get_all_dtypes(include_bool=False, include_half=False,
                                          include_bfloat16=False, include_complex=False))
    def test_factory_device_type_inference(self, device, dtype):
        cpu_cuda = ('cpu', 'cuda')
        cpu_cuda_none = cpu_cuda + (None,)
        for crow_indices_device, col_indices_device, values_device, device in itertools.product(cpu_cuda,
                                                                                                cpu_cuda,
                                                                                                cpu_cuda,
                                                                                                cpu_cuda_none):
            for index_dtype in [torch.int32, torch.int64]:
                crow_indices = torch.tensor([0, 2, 4], dtype=index_dtype, device=crow_indices_device)
                col_indices = torch.tensor([0, 1, 0, 1], dtype=index_dtype, device=col_indices_device)
                values = torch.tensor([1, 2, 3, 4], dtype=dtype, device=values_device)
                if device is None and (crow_indices_device != col_indices_device or
                                       crow_indices_device != values_device):
                    with self.assertRaises(RuntimeError):
                        torch.sparse_csr_tensor(crow_indices,
                                                col_indices,
                                                values,
                                                size=(2, 10),
                                                device=device)
                else:
                    t = torch.sparse_csr_tensor(crow_indices,
                                                col_indices,
                                                values,
                                                size=(2, 10),
                                                device=device)
                    should_be_cuda = (device == 'cuda' or (device is None and values_device == 'cuda'))
                    self.assertEqual(should_be_cuda, t.is_cuda)
                    t.crow_indices().dtype == index_dtype
                    t.col_indices().dtype == index_dtype
                    t.values().dtype == dtype
                    t.crow_indices().device == t.values().device
                    t.col_indices().device == t.values().device

    def test_sparse_csr_print(self, device):
        orig_maxDiff = self.maxDiff
        self.maxDiff = None
        shape_nnz = [
            ((10, 10), 10),
            ((100, 10), 10),
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
            for index_dtype in [torch.int32, torch.int64]:
                for dtype in torch.testing.floating_types():
                    printed.append("########## {}/{} ##########".format(dtype, index_dtype))
                    x = torch.sparse_csr_tensor(torch.tensor([0, 2, 4], dtype=index_dtype),
                                                torch.tensor([0, 1, 0, 1], dtype=index_dtype),
                                                torch.tensor([1, 2, 3, 4]), dtype=dtype, device=device)
                    printed.append("# sparse tensor")
                    printed.append(str(x))
                    printed.append("# _crow_indices")
                    printed.append(str(x.crow_indices()))
                    printed.append("# _col_indices")
                    printed.append(str(x.col_indices()))
                    printed.append("# _values")
                    printed.append(str(x.values()))
                    printed.append('')
                printed.append('')
        self.assertExpected('\n'.join(printed))
        self.maxDiff = orig_maxDiff

    def test_sparse_csr_from_dense(self, device):
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

    @coalescedonoff
    @dtypes(torch.double)
    def test_coo_to_csr_convert(self, device, dtype, coalesced):
        size = (5, 5)
        sparse_dim = 2
        nnz = 10
        sparse_coo, _, _ = self.genSparseTensor(size, sparse_dim, nnz, coalesced, device, dtype)
        sparse_csr = sparse_coo.to_sparse_csr()

        self.assertTrue(sparse_csr.is_sparse_csr)
        self.assertEqual(sparse_csr.to_dense(), sparse_coo.to_dense())

        vec = torch.randn((5, 1), dtype=dtype, device=device)
        coo_product = sparse_coo.matmul(vec)
        csr_product = sparse_csr.matmul(vec)

        self.assertEqual(coo_product, csr_product)

        vec = torch.randn((100, 1), dtype=dtype, device=device)
        index = torch.tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], dtype=torch.int32)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        coo = torch.sparse_coo_tensor(index, values, torch.Size([100, 100]), dtype=dtype, device=device)
        csr = coo.to_sparse_csr()

        self.assertEqual(coo.matmul(vec), csr.matmul(vec))

    @onlyCPU
    @unittest.skipIf(IS_MACOS or IS_WINDOWS, "MKL doesn't work on windows or mac")
    @dtypes(torch.float, torch.double)
    def test_mkl_matvec_warnings(self, device, dtype):
        if torch.has_mkl:
            for index_dtype in [torch.int32, torch.int64]:
                sp = torch.sparse_csr_tensor(torch.tensor([0, 2, 4]),
                                             torch.tensor([0, 1, 0, 1]),
                                             torch.tensor([1, 2, 3, 4], dtype=dtype, device=device))
                vec = torch.randn((2, 1), dtype=dtype, device=device)
                with warnings.catch_warnings(record=True) as w:
                    sp.matmul(vec)
                    self.assertEqual(len(w), 2)
                    self.assertIn("Pytorch is compiled with MKL LP64 and will convert crow_indices to int32",
                                  str(w[0].message))
                    self.assertIn("Pytorch is compiled with MKL LP64 and will convert col_indices to int32",
                                  str(w[1].message))

    @dtypes(torch.double)
    def test_dense_convert_error(self, device, dtype):
        size = (4, 2, 4)
        dense = torch.randn(size, dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, "Only 2D"):
            sparse = dense.to_sparse_csr()

    # TODO: Support auto generation of device check for sparse tensors
    # See: https://github.com/pytorch/pytorch/issues/59058
    @dtypes(torch.double)
    def test_matmul_device_mismatch(self, device, dtype):
        cpu = torch.rand((10, 10))
        cuda = cpu.cuda()
        for s, m1, m2 in itertools.product((cpu, cuda), repeat=3):
            csr = m1.to_sparse()
            if s.device == csr.device == m2.device:
                torch.addmm(s, csr, m2)
            else:
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    torch.addmm(s, csr, m2)

    @dtypes(torch.float, torch.double)
    def test_csr_matvec(self, device, dtype):
        side = 100
        for index_dtype in [torch.int32, torch.int64]:
            csr = self.genSparseCSRTensor((side, side), 1000, device=device, dtype=dtype, index_dtype=index_dtype)
            vec = torch.randn(side, dtype=dtype, device=device)

            res = csr.matmul(vec)
            expected = csr.to_dense().matmul(vec)

            self.assertEqual(res, expected)

            bad_vec = torch.randn(side + 10, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, "mv: expected"):
                csr.matmul(bad_vec)

    @dtypes(torch.double)
    def test_mm(self, device, dtype):
        def test_shape(di, dj, dk, nnz):
            for index_dtype in [torch.int32, torch.int64]:
                x = self.genSparseCSRTensor((di, dj), nnz, device=device, dtype=dtype, index_dtype=index_dtype)
                t = torch.randn(di, dk, dtype=dtype, device=device)
                y = torch.randn(dj, dk, dtype=dtype, device=device)
                alpha = random.random()
                beta = random.random()

                # res = beta * t  + alpha * (x @ y)
                res = torch.addmm(t, x, y, beta=beta, alpha=alpha)
                expected = torch.addmm(t, x.to_dense(), y, beta=beta, alpha=alpha)
                self.assertEqual(res, expected)

                res = torch.addmm(t, x, y)
                expected = torch.addmm(t, x.to_dense(), y)
                self.assertEqual(res, expected)

                res = torch.mm(x, y)
                expected = torch.mm(x.to_dense(), y)
                self.assertEqual(res, expected)

        for i in range(2, 5):
            for j in range(2, 8):
                for k in range(2, 8):
                    test_shape(i, j, k, i * j // 2)
        test_shape(4, 4, 4, 0)

    @dtypes(*torch.testing.floating_types())
    def test_sparse_mm(self, device, dtype):
        def test_shape(d1, d2, d3, nnz, transposed):
            if transposed:
                D = torch.randn(d3, d2, dtype=dtype, device=device).t_()
            else:
                D = torch.randn(d2, d3, dtype=dtype, device=device)
            S = self.genSparseCSRTensor((d1, d2), nnz, device=device, dtype=dtype, index_dtype=torch.int32)
            S_dense = S.to_dense()
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

        test_shape(7, 8, 9, 20, False)
        test_shape(7, 8, 9, 20, True)

    @dtypes(*torch.testing.floating_types())
    def test_sparse_addmm(self, device, dtype):
        def test_shape(m, n, p, nnz, broadcast, alpha_beta=None):
            if alpha_beta is None:
                alpha = random.random()
                beta = random.random()
            else:
                alpha, beta = alpha_beta
            if broadcast:
                D1 = make_tensor((), dtype=dtype, device=device)
            else:
                D1 = make_tensor([n, p], dtype=dtype, device=device)
            D2 = make_tensor([m, p], dtype=dtype, device=device)
            S = self.genSparseCSRTensor([n, m], nnz, dtype=dtype, device=device, index_dtype=torch.int32)
            S_dense = S.to_dense()
            Y = torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            Y_dense = torch.addmm(D1, S_dense, D2, beta=beta, alpha=alpha)
            self.assertEqual(Y, Y_dense)

        test_shape(7, 8, 9, 20, False, None)
        test_shape(7, 8, 9, 20, True, None)
        test_shape(7, 8, 9, 20, False, (1, 0))
        test_shape(7, 8, 9, 20, True, (1, 0))
        test_shape(7, 8, 9, 20, False, (1, 1))
        test_shape(7, 8, 9, 20, True, (1, 1))

    @dtypes(torch.float, torch.double)
    def test_add(self, device, dtype):
        def _test_spadd_shape(nnz, shape):
            x = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=torch.int32)
            y = torch.randn(*shape, dtype=dtype, device=device)
            r = random.random()

            res = torch.add(y, x, alpha=r)
            expected = y + r * x.to_dense()
            self.assertEqual(res, expected)

            # Non contiguous dense tensor
            s = list(shape)
            s[0] = shape[-1]
            s[-1] = shape[0]
            y = torch.randn(*s, dtype=torch.double, device=device)
            y.transpose_(0, len(s) - 1)
            r = random.random()

            res = torch.add(y, x, alpha=r)
            expected = y + r * x.to_dense()

            self.assertEqual(res, expected)

        _test_spadd_shape(10, [100, 100])
        _test_spadd_shape(0, [100, 100])
        _test_spadd_shape(10, [100, 1])
        _test_spadd_shape(10, [1, 100])

    @dtypes(*torch.testing.floating_types())
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
