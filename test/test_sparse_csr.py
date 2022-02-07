# Owner(s): ["module: sparse"]

import torch
import random
import itertools
import unittest
from torch.testing import get_all_complex_dtypes, get_all_fp_dtypes, floating_and_complex_types, make_tensor
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC
from torch.testing._internal.common_utils import \
    (TEST_WITH_ROCM, TEST_SCIPY, TestCase, run_tests, load_tests, coalescedonoff)
from torch.testing._internal.common_device_type import \
    (ops, instantiate_device_type_tests, dtypes, dtypesIfCUDA, onlyCPU, onlyCUDA, skipCUDAIfNoCusparseGeneric,
     precisionOverride, skipMeta, skipCUDAIf, skipCUDAIfRocm, skipCPUIfNoMklSparse)
from torch.testing._internal.common_methods_invocations import \
    (op_db, sparse_csr_unary_ufuncs, )
from torch.testing._internal.common_cuda import _get_torch_cuda_version, CUDA11OrLater
from torch.testing._internal.common_dtype import floating_types, get_all_dtypes
from test_sparse import CUSPARSE_SPMM_COMPLEX128_SUPPORTED

if TEST_SCIPY:
    import scipy.sparse as sp

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

def _check_cusparse_triangular_solve_available():
    version = _get_torch_cuda_version()
    # cusparseSpSM was added in 11.3.1 but we don't have access to patch version
    min_supported_version = (11, 4)
    return version >= min_supported_version

def _check_cusparse_spgemm_available():
    # cusparseSpGEMM was added in 11.0
    version = _get_torch_cuda_version()
    min_supported_version = (11, 0)
    return version >= min_supported_version

def _check_cusparse_sddmm_available():
    version = _get_torch_cuda_version()
    # cusparseSDDMM was added in 11.2.1 but we don't have access to patch version
    min_supported_version = (11, 3)
    return version >= min_supported_version

_sparse_csr_ops = list(filter(lambda op: op.supports_sparse_csr, op_db))

# This should be just an import from test_linalg instead of code duplication
# but https://github.com/pytorch/pytorch/pull/63511#discussion_r733989701
def _test_addmm_addmv(test_case, f, t, m, v, *, alpha=None, beta=None, transpose_out=False, layout=torch.strided, all_sparse=False):
    """
    Unified test for checking `f(t, m, v, alpha=alpha, beta=beta)` computation,
    where f is `torch.addmv` or `torch.addmm`.
    `transpose_out` controls whether the out argument is in column-major order.
    `layout` controls whether `m` is converted to specified layout or not.
    Custom behaviour is implemented only for torch.sparse_csr layout.
    """
    dtype = t.dtype
    numpy_dtype = dtype
    if dtype in {torch.bfloat16}:
        numpy_dtype = torch.float
    if dtype.is_complex:
        alpha = 0.9 + 0.3j if alpha is None else alpha
        beta = 0.5 + 0.6j if beta is None else beta
    else:
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta

    def convert_layout(mat):
        if layout == torch.sparse_csr:
            return mat.to_sparse_csr()
        else:
            assert mat.layout == layout
            return mat

    if all_sparse:
        res1 = f(*map(convert_layout, (t, m, v)), alpha=alpha, beta=beta)
        res1 = res1.to_dense()
    else:
        res1 = f(t, convert_layout(m), v, alpha=alpha, beta=beta)
    res2 = torch.full_like(res1, float('nan'))
    if transpose_out:
        res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
    f(t, convert_layout(m), v, alpha=alpha, beta=beta, out=res2)
    res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
    if beta != 0:
        res3 += (beta * t).to(numpy_dtype).cpu().numpy()
    res3 = torch.from_numpy(res3).to(dtype)
    test_case.assertEqual(res1, res2)
    test_case.assertEqual(res1, res3)


class TestSparseCSRSampler(TestCase):

    def test_make_crow_indices(self):
        # Here we test the correctness of the crow_indices algorithm
        # and testing it on CPU and with int32 dtype will be
        # sufficient.
        device = torch.device('cpu')
        index_dtype = torch.int32
        for n_rows in range(1, 10):
            for n_cols in range(1, 10):
                for nnz in range(0, n_rows * n_cols + 1):
                    crow_indices = self._make_crow_indices(
                        n_rows, n_cols, nnz,
                        device=device, dtype=index_dtype)
                    self.assertEqual(len(crow_indices), n_rows + 1)
                    counts = crow_indices[1:] - crow_indices[:-1]
                    self.assertEqual(counts.sum(), nnz)
                    self.assertGreaterEqual(counts.min(), 0)
                    self.assertLessEqual(counts.max(), n_cols)


class TestSparseCSR(TestCase):

    @onlyCPU
    def test_csr_layout(self):
        self.assertEqual(str(torch.sparse_csr), 'torch.sparse_csr')
        self.assertEqual(type(torch.sparse_csr), torch.layout)

    @dtypes(*get_all_dtypes())
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

    @dtypes(*get_all_dtypes())
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

    @dtypes(*get_all_dtypes())
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

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_empty(self, device, dtype):
        ns = [5, 2, 0]
        for shape in itertools.product(ns, ns):
            result = torch.empty(shape, dtype=dtype, device=device, layout=torch.sparse_csr)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, torch.device(device))
            self.assertEqual(result.layout, torch.sparse_csr)
            self.assertEqual(result.crow_indices().shape, (shape[0] + 1,))
            self.assertEqual(result.col_indices().shape, (0,))
            self.assertEqual(result.values().shape, (0,))
            self.assertEqual(result._nnz(), 0)
            self.assertEqual(result.crow_indices().device, torch.device(device))
            self.assertEqual(result.col_indices().device, torch.device(device))
            self.assertEqual(result.values().device, torch.device(device))
            self.assertEqual(result.crow_indices().dtype, torch.int64)
            self.assertEqual(result.col_indices().dtype, torch.int64)
            self.assertEqual(result.values().dtype, dtype)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_empty_errors(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError, "torch.empty: Only 2D sparse CSR tensors are supported."):
            torch.empty((5,), dtype=dtype, device=device, layout=torch.sparse_csr)

        with self.assertRaisesRegex(RuntimeError, "torch.empty: Only 2D sparse CSR tensors are supported."):
            torch.empty((2, 3, 4), dtype=dtype, device=device, layout=torch.sparse_csr)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_clone(self, device, dtype):
        x = torch.sparse_csr_tensor([0, 2, 4],
                                    [0, 1, 0, 1],
                                    [1, 2, 3, 4],
                                    dtype=dtype,
                                    device=device)
        y = x.clone()

        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.crow_indices(), y.crow_indices())
        self.assertEqual(x.col_indices(), y.col_indices())
        self.assertEqual(x.values(), y.values())

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_copy(self, device, dtype):

        def run_test(shape, nnz, index_type):
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            b = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            a.copy_(b)

            self.assertEqual(a.crow_indices(), b.crow_indices())
            self.assertEqual(a.col_indices(), b.col_indices())
            self.assertEqual(a.values(), b.values())

        ns = [5, 2, 0]
        for shape, index_dtype in zip(itertools.product(ns, ns), [torch.int32, torch.int64]):
            run_test(shape, 0, index_dtype)
            run_test(shape, shape[0] * shape[1], index_dtype)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_copy_errors(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            shape1 = (2, 3)
            shape2 = (3, 2)
            a = self.genSparseCSRTensor(shape1, 0, dtype=dtype, device=device, index_dtype=index_dtype)
            b = self.genSparseCSRTensor(shape2, 0, dtype=dtype, device=device, index_dtype=index_dtype)

            with self.assertRaisesRegex(RuntimeError, "only same size tensors are supported."):
                a.copy_(b)

            with self.assertRaisesRegex(RuntimeError, "copy between different layouts is not supported."):
                a.copy_(torch.empty(a.shape, dtype=dtype, device=device))

            b = self.genSparseCSRTensor(shape1, 1, dtype=dtype, device=device, index_dtype=index_dtype)
            with self.assertRaisesRegex(RuntimeError, "only tensors with the same number of specified elements are supported."):
                a.copy_(b)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_resize(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            shape = (2, 3)
            nnz = 6
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            new_shape = (4, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to larger shape doesn't add specified elements
            self.assertEqual(a._nnz(), nnz)

            new_shape = (1, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to smaller shape trims specified elements
            self.assertEqual(a._nnz(), 5)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_resize_errors(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            shape = (2, 3)
            nnz = 6
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            with self.assertRaisesRegex(RuntimeError, "torch.resize_: Only 2D sparse CSR tensors are supported."):
                new_shape = (4,)
                a.resize_(new_shape)

            # resizing of columns to smaller size is not implemented
            with self.assertRaisesRegex(
                RuntimeError,
                "torch.resize_: Resizing columns of sparse CSR tensors to a smaller value is not supported.",
            ):
                new_shape = (2, 2)
                a.resize_(new_shape)

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
    @dtypes(*get_all_dtypes())
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
                for dtype in floating_types():
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

    @dtypes(*get_all_dtypes())
    def test_sparse_csr_from_dense(self, device, dtype):
        dense = torch.tensor([[4, 5, 0], [0, 0, 0], [1, 0, 0]], dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 2, 3], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([4, 5, 1], dtype=dtype), sparse.values())

        dense = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 0, 1, 2], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([2, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([1, 1], dtype=dtype), sparse.values())

        dense = torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 3, 6, 9], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 2] * 3, dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([2] * 9, dtype=dtype), sparse.values())

    @dtypes(*get_all_dtypes())
    def test_sparse_csr_to_dense(self, device, dtype):
        mn = [5, 2, 0]
        for (m, n) in itertools.product(mn, mn):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            sparse = dense.to_sparse_csr()
            self.assertEqual(sparse.to_dense(), dense)

        crow_indices = torch.tensor([0, 3, 5])
        col_indices = torch.tensor([0, 1, 2, 0, 1])
        values = torch.tensor([1, 2, 1, 3, 4], dtype=dtype)
        csr = torch.sparse_csr_tensor(crow_indices, col_indices,
                                      values, dtype=dtype, device=device)
        dense = torch.tensor([[1, 2, 1], [3, 4, 0]], dtype=dtype, device=device)
        self.assertEqual(csr.to_dense(), dense)

    @skipMeta
    @skipCPUIfNoMklSparse
    @coalescedonoff
    @dtypes(torch.double)
    def test_coo_to_csr_convert(self, device, dtype, coalesced):
        with self.assertRaisesRegex(RuntimeError, "Input is supposed to be a vector"):
            torch._convert_indices_from_coo_to_csr(
                torch.randint(100, (5, 5), device=device),
                size=100)

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

        col_indices = torch.tensor([
            31, 92, 65, 50, 34, 62, 22, 56, 74, 89
        ], dtype=torch.int64, device=device)
        self.assertEqual(csr.col_indices(), col_indices)

        values = torch.tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7], dtype=dtype, device=device)
        self.assertEqual(csr.values(), values)

    @dtypes(*get_all_dtypes())
    def test_sparse_csr_from_dense_convert_error(self, device, dtype):
        size = (4, 2, 4)
        dense = make_tensor(size, dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, "Only 2D"):
            sparse = dense.to_sparse_csr()

    # TODO: Support auto generation of device check for sparse tensors
    # See: https://github.com/pytorch/pytorch/issues/59058
    @onlyCUDA
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

    @skipCPUIfNoMklSparse
    @skipCUDAIfNoCusparseGeneric
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*get_all_complex_dtypes(),
                  *get_all_fp_dtypes(include_half=SM53OrLater, include_bfloat16=SM80OrLater))
    def test_csr_matvec(self, device, dtype):
        side = 100
        for index_dtype in [torch.int32, torch.int64]:
            csr = self.genSparseCSRTensor((side, side), 1000, device=device, dtype=dtype, index_dtype=index_dtype)
            vec = torch.randn(side, dtype=dtype, device=device)

            res = csr.matmul(vec)
            expected = csr.to_dense().matmul(vec)

            self.assertEqual(res, expected)

            bad_vec = torch.randn(side + 10, dtype=dtype, device=device)
            err_msg = "size mismatch, got"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                csr.matmul(bad_vec)

    @onlyCUDA
    @unittest.skipIf(not CUDA11OrLater, "Only CUDA 11+ is supported")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_baddbmm(self, device, dtype):
        def run_test(c, a, a_batched, b, op_b=False, op_out=False, *, dtype=None, device=None):
            alpha = complex(random.random(), random.random()) if dtype.is_complex else random.random()
            beta = complex(random.random(), random.random()) if dtype.is_complex else random.random()
            b = b.mH if (op_b and a.shape == b.shape) else b

            actual = torch.baddbmm(c, a_batched, b, alpha=alpha, beta=beta)

            out = torch.empty_like(c.mH if op_out and a.shape == b.shape else c)
            torch.baddbmm(c, a_batched, b, alpha=alpha, beta=beta, out=out)

            expected = [torch.addmm(c[i], a, b[i], alpha=alpha, beta=beta) for i in range(c.shape[0])]
            expected = torch.stack(expected, 0)

            self.assertEqual(actual, out)
            self.assertEqual(actual, expected)

        for index_dtype in [torch.int32, torch.int64]:
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([1, 5], repeat=3), [1, 3], [True, False]):
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)

                # a_batched is a regular CSR tensor but with a batch dimension in the shape
                a_batched = torch._sparse_csr_tensor_unsafe(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k))

                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                c = make_tensor((batch_size, m, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    run_test(c, a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    @onlyCUDA
    @unittest.skipIf(not CUDA11OrLater, "Only CUDA 11+ is supported")
    @skipCUDAIfNoCusparseGeneric
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_bmm(self, device, dtype):
        def run_test(a, a_batched, b, op_b=False, op_out=False, *, dtype=None, device=None):
            b = b.mH if (op_b and a.shape == b.shape) else b

            actual = torch.bmm(a_batched, b)

            out = torch.empty_like(actual.mH if op_out and a.shape == b.shape else actual)
            torch.bmm(a_batched, b, out=out)

            expected = [torch.mm(a, b[i]) for i in range(b.shape[0])]
            expected = torch.stack(expected, 0)

            self.assertEqual(actual, out)
            self.assertEqual(actual, expected)

        for index_dtype in [torch.int32, torch.int64]:
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([1, 5], repeat=3), [1, 3], [True, False]):
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)

                # a_batched is a regular CSR tensor but with a batch dimension in the shape
                a_batched = torch._sparse_csr_tensor_unsafe(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k))

                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    run_test(a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    def run_test_block_addmm_addmv(self, addmv_addmm, c, a, b, op_b=False, op_out=False, *, dtype=None, device=None):
        alpha = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        beta = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        b = b.mH if (op_b and a.shape == b.shape) else b

        actual = addmv_addmm(c, a, b, alpha=alpha, beta=beta)

        out = torch.empty_like(c.mH if op_out and a.shape == b.shape else c)
        addmv_addmm(c, a, b, alpha=alpha, beta=beta, out=out)

        a_bsr = sp.bsr_matrix(
            (
                a.values().cpu().numpy(),
                a.col_indices().cpu().numpy(),
                a.crow_indices().cpu().numpy(),
            ),
            shape=a.shape,
        )
        expected = alpha * (a_bsr * b.cpu().resolve_conj().numpy()) + beta * c.cpu().numpy()
        self.assertEqual(actual, out)
        self.assertEqual(actual, expected)

    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_addmm(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            for (m, n, k), block_size, noncontiguous in zip(itertools.product([1, 5], repeat=3), [1, 2, 3], [True, False]):
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data   # Test column-major blocks
                a = torch._sparse_csr_tensor_unsafe(a.crow_indices(), a.col_indices(), a_data, (m * block_size, k * block_size))
                b = make_tensor((k * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
                c = make_tensor((m * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    self.run_test_block_addmm_addmv(torch.addmm, c, a, b, op_b, op_out, dtype=dtype, device=device)

    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_addmv(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            block_sizes = [1, 2, 3]
            if TEST_WITH_ROCM or not TEST_CUSPARSE_GENERIC:
                block_sizes = [2, 3]
            for (m, k), block_size, noncontiguous in zip(itertools.product([1, 5], repeat=2), block_sizes, [True, False]):
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data  # Test column-major blocks
                a = torch._sparse_csr_tensor_unsafe(a.crow_indices(), a.col_indices(), a_data, (m * block_size, k * block_size))
                b = make_tensor((k * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
                c = make_tensor((m * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
                self.run_test_block_addmm_addmv(torch.addmv, c, a, b, dtype=dtype, device=device)

    @skipCPUIfNoMklSparse
    @skipCUDAIfRocm
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_triangular_solve(self, device, dtype):
        def run_test(a, b, upper, transpose, unitriangular, op_out):
            actual = torch.triangular_solve(b, a, upper=upper, unitriangular=unitriangular, transpose=transpose)
            actual_X = actual.solution
            actual_A_clone = actual.cloned_coefficient
            self.assertTrue(actual_A_clone.numel() == 0)
            if a._nnz() == 0:
                self.assertTrue(actual_X.isnan().all())
                return

            # TODO: replace with torch method when implemented to_dense() on block sparse tensor
            a_bsr = sp.bsr_matrix(
                (
                    a.values().cpu().numpy(),
                    a.col_indices().cpu().numpy(),
                    a.crow_indices().cpu().numpy(),
                ),
                shape=a.shape,
            )
            expected_X, _ = torch.triangular_solve(
                b,
                torch.tensor(a_bsr.todense(), device=device),
                transpose=transpose,
                upper=upper,
                unitriangular=unitriangular)
            self.assertEqual(actual_X, expected_X)

            out = torch.empty_like(b.mH if op_out and a.shape == b.shape else b)
            torch.triangular_solve(
                b, a,
                upper=upper, unitriangular=unitriangular, transpose=transpose, out=(out, actual_A_clone)
            )
            self.assertEqual(out, actual_X)
            self.assertEqual(out, expected_X)

        for index_dtype in [torch.int32, torch.int64]:
            for (m, k), block_size, noncontiguous in zip(itertools.product([1, 5], repeat=2), [2, 3], [True, False]):
                nnz = random.randint(0, m * m)
                a = self.genSparseCSRTensor((m, m), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data  # Test column-major blocks
                a = torch._sparse_csr_tensor_unsafe(a.crow_indices(), a.col_indices(), a_data, (m * block_size, m * block_size))
                b = make_tensor((m * block_size, k), dtype=dtype, device=device, noncontiguous=noncontiguous)

                for (upper, unitriangular, transpose, op_out) in itertools.product([True, False], repeat=4):
                    run_test(a, b, upper, unitriangular, transpose, op_out)

    @skipCPUIfNoMklSparse
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

    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*get_all_complex_dtypes(),
                  *get_all_fp_dtypes(include_half=SM53OrLater and TEST_CUSPARSE_GENERIC,
                                     include_bfloat16=SM80OrLater and TEST_CUSPARSE_GENERIC))
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_sparse_mm(self, device, dtype):
        def test_shape(d1, d2, d3, nnz, transposed, index_dtype):
            if transposed:
                D = torch.randn(d3, d2, dtype=dtype, device=device).t_()
            else:
                D = torch.randn(d2, d3, dtype=dtype, device=device)
            S = self.genSparseCSRTensor((d1, d2), nnz, device=device, dtype=dtype, index_dtype=index_dtype)
            S_dense = S.to_dense()
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

        for index_dtype in [torch.int32, torch.int64]:
            test_shape(7, 8, 9, 20, False, index_dtype)
            test_shape(7, 8, 9, 20, True, index_dtype)

    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*get_all_complex_dtypes(),
                  *get_all_fp_dtypes(include_half=SM53OrLater and TEST_CUSPARSE_GENERIC,
                                     include_bfloat16=SM80OrLater and TEST_CUSPARSE_GENERIC))
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_sparse_addmm(self, device, dtype):
        def test_shape(m, n, p, nnz, broadcast, index_dtype, alpha_beta=None):
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
            S = self.genSparseCSRTensor([n, m], nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            S_dense = S.to_dense()
            Y = torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            Y_dense = torch.addmm(D1, S_dense, D2, beta=beta, alpha=alpha)
            self.assertEqual(Y, Y_dense)

        for index_dtype in [torch.int32, torch.int64]:
            test_shape(7, 8, 9, 20, False, index_dtype, None)
            test_shape(7, 8, 9, 20, True, index_dtype, None)
            test_shape(7, 8, 9, 20, False, index_dtype, (1, 0))
            test_shape(7, 8, 9, 20, True, index_dtype, (1, 0))
            test_shape(7, 8, 9, 20, False, index_dtype, (1, 1))
            test_shape(7, 8, 9, 20, True, index_dtype, (1, 1))

    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(torch.complex64,
                  *((torch.complex128,) if CUSPARSE_SPMM_COMPLEX128_SUPPORTED else ()),
                  *torch.testing.get_all_fp_dtypes(include_bfloat16=SM80OrLater,
                                                   include_half=SM53OrLater))
    @skipCUDAIf(
        not _check_cusparse_spgemm_available(),
        "cuSparse Generic API SpGEMM is not available"
    )
    def test_addmm_all_sparse_csr(self, device, dtype):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, all_sparse=True)

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, all_sparse=True)

        # Test beta=0, M=nan
        M = torch.full((10, 25), float('nan'), device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, beta=0, layout=torch.sparse_csr, all_sparse=True)

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            _test_addmm_addmv(self, torch.addmm, M, m1, m2, transpose_out=t4, layout=torch.sparse_csr, all_sparse=True)

    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(torch.complex64,
                  *((torch.complex128,) if CUSPARSE_SPMM_COMPLEX128_SUPPORTED else ()),
                  *torch.testing.get_all_fp_dtypes(include_bfloat16=SM80OrLater,
                                                   include_half=SM53OrLater))
    @skipCUDAIf(
        not _check_cusparse_spgemm_available(),
        "cuSparse Generic API SpGEMM is not available"
    )
    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    def test_addmm_sizes_all_sparse_csr(self, device, dtype):
        for m in [0, 1, 25]:
            for n in [0, 1, 10]:
                for k in [0, 1, 8]:
                    M = torch.randn(n, m, device=device).to(dtype)
                    m1 = torch.randn(n, k, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, all_sparse=True)

                    M = torch.randn(n, m, device=device).to(dtype).to_sparse_csr()
                    m1 = torch.randn(n, k + 1, device=device).to(dtype).to_sparse_csr()
                    m2 = torch.randn(k, m, device=device).to(dtype).to_sparse_csr()
                    self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.addmm(M, m1, m2))
                    self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.mm(m1, m2))

    @skipCPUIfNoMklSparse
    @dtypes(torch.float)
    def test_addmm_errors(self, device, dtype):
        # test that the errors are the same for dense and sparse versions
        import re

        def test1(*, is_sparse):
            # shapes must be compatible for matrix multiplication
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.addmm(a, a_sparse, a)
            else:
                return torch.addmm(a, a, a)

        def test2(*, is_sparse):
            # mat2 must be a matrix
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.addmm(a, a_sparse, a.unsqueeze(0))
            else:
                return torch.addmm(a, a, a.unsqueeze(0))

        def test3(*, is_sparse):
            # the first input needs to be 1D or 2D
            a = make_tensor((3, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.addmm(a.unsqueeze(0), a_sparse, a)
            else:
                return torch.addmm(a.unsqueeze(0), a, a)

        for test in (test1, test2, test3):
            try:
                test(is_sparse=False)
            except RuntimeError as msg:
                with self.assertRaisesRegex(RuntimeError, re.escape(str(msg))):
                    test(is_sparse=True)

    @skipCPUIfNoMklSparse
    @dtypes(torch.float)
    def test_mm_errors(self, device, dtype):
        # test that the errors are the same for dense and sparse versions
        import re

        def test1(*, is_sparse):
            # shapes must be compatible for matrix multiplication
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.mm(a_sparse, a)
            else:
                return torch.mm(a, a)

        def test2(*, is_sparse):
            # mat2 must be a matrix
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.mm(a_sparse, a.unsqueeze(0))
            else:
                return torch.mm(a, a.unsqueeze(0))

        for test in (test1, test2):
            try:
                test(is_sparse=False)
            except RuntimeError as msg:
                with self.assertRaisesRegex(RuntimeError, re.escape(str(msg))):
                    test(is_sparse=True)

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

    @skipCPUIfNoMklSparse
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_sparse_add(self, device, dtype):
        def run_test(m, n, index_dtype):

            if TEST_WITH_ROCM and dtype.is_complex:
                self.skipTest("ROCm doesn't work with complex dtype correctly.")

            alpha = random.random()
            nnz1 = random.randint(0, m * n)
            nnz2 = random.randint(0, m * n)
            nnz3 = random.randint(0, m * n)

            if TEST_WITH_ROCM:
                # ROCm fails when nnz = 0
                nnz1, nnz2, nnz3 = max(1, nnz1), max(1, nnz2), max(1, nnz3)

            S1 = self.genSparseCSRTensor([m, n], nnz1, dtype=dtype, device=device, index_dtype=index_dtype)
            S2 = self.genSparseCSRTensor([m, n], nnz2, dtype=dtype, device=device, index_dtype=index_dtype)
            S3 = self.genSparseCSRTensor([m, n], nnz3, dtype=dtype, device=device, index_dtype=index_dtype)

            expected = torch.add(S1.to_dense(), S2.to_dense(), alpha=alpha)
            actual = torch.add(S1, S2, alpha=alpha, out=S3)

            self.assertEqual(actual.to_dense(), expected)
            self.assertEqual(S3.to_dense(), expected)

        for index_dtype in [torch.int32, torch.int64]:
            for m, n in itertools.product([3, 5], [3, 5]):
                run_test(m, n, index_dtype)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_sparse_add_errors(self, device, dtype):
        def run_test(index_type):
            a = self.genSparseCSRTensor((2, 2), 3, dtype=dtype, device=device, index_dtype=index_dtype)
            b = self.genSparseCSRTensor((2, 1), 2, dtype=dtype, device=device, index_dtype=index_dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected input tensors to have the same shape"):
                torch.add(a, b)

        for index_dtype in [torch.int32, torch.int64]:
            run_test(index_dtype)

    @skipCPUIfNoMklSparse
    @skipCUDAIf(
        not _check_cusparse_triangular_solve_available(),
        "cuSparse Generic API SpSV is not available"
    )
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_sparse_triangular_solve(self, device, dtype):

        def run_test(n, k, upper, unitriangular, transpose, zero):
            triangle_function = torch.triu if upper else torch.tril
            make_A = torch.zeros if zero else make_tensor
            A = make_A((n, n), dtype=dtype, device=device)
            A = triangle_function(A)
            A_sparse = A.to_sparse_csr()
            B = make_tensor((n, k), dtype=dtype, device=device)

            expected = torch.triangular_solve(B, A, upper=upper, unitriangular=unitriangular, transpose=transpose)
            expected_X = expected.solution

            actual = torch.triangular_solve(B, A_sparse, upper=upper, unitriangular=unitriangular, transpose=transpose)
            actual_X = actual.solution
            actual_A_clone = actual.cloned_coefficient
            self.assertTrue(actual_A_clone.numel() == 0)
            if A_sparse._nnz() == 0:
                self.assertTrue(actual_X.isnan().all())
                return
            self.assertEqual(actual_X, expected_X)

            # test out with C contiguous strides
            out = torch.empty_strided((n, k), (k, 1), dtype=dtype, device=device)
            torch.triangular_solve(
                B, A_sparse,
                upper=upper, unitriangular=unitriangular, transpose=transpose, out=(out, actual_A_clone)
            )
            self.assertEqual(out, expected_X)

            # test out with F contiguous strides
            out = torch.empty_strided((n, k), (1, n), dtype=dtype, device=device)
            torch.triangular_solve(
                B, A_sparse,
                upper=upper, unitriangular=unitriangular, transpose=transpose, out=(out, actual_A_clone)
            )
            self.assertEqual(out, expected_X)
            self.assertEqual(out.stride(), (1, n))

            # test out with discontiguous strides
            out = torch.empty_strided((2 * n, k), (1, 2 * n), dtype=dtype, device=device)[::2]
            if n > 0 and k > 0:
                self.assertFalse(out.is_contiguous())
                self.assertFalse(out.t().is_contiguous())
            before_stride = out.stride()
            torch.triangular_solve(
                B, A_sparse,
                upper=upper, unitriangular=unitriangular, transpose=transpose, out=(out, actual_A_clone)
            )
            self.assertEqual(out, expected_X)
            self.assertEqual(out.stride(), before_stride)

        ks = [0, 1, 3]
        ns = [5, 3, 0]
        for (k, n), (upper, unitriangular, transpose, zero) in itertools.product(itertools.product(ks, ns),
                                                                                 itertools.product([True, False], repeat=4)):
            run_test(n, k, upper, unitriangular, transpose, zero)

    @skipCUDAIfRocm
    @onlyCUDA
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_sampled_addmm(self, device, dtype):
        def run_test(c, a, b, op_a, op_b, *, alpha=None, beta=None):
            if dtype.is_complex:
                alpha = random.random() + 0.3j if alpha is None else alpha
                beta = random.random() + 0.6j if beta is None else beta
            else:
                alpha = random.random() if alpha is None else alpha
                beta = random.random() if beta is None else beta

            if op_a and a.shape == b.shape:
                a = a.mH
            if op_b and a.shape == b.shape:
                b = b.mH

            actual = torch.sparse.sampled_addmm(c, a, b, alpha=alpha, beta=beta)

            out = torch.sparse_csr_tensor(
                *map(torch.clone, (actual.crow_indices(), actual.col_indices())),
                torch.empty_like(actual.values()),
                size=c.shape
            )
            torch.sparse.sampled_addmm(c, a, b, alpha=alpha, beta=beta, out=out)

            spy_c = torch.sparse_csr_tensor(c.crow_indices(), c.col_indices(), torch.ones_like(c.values()), size=c.shape)
            expected = alpha * (a @ b) * spy_c.to_dense() + beta * c.to_dense()
            self.assertEqual(actual.to_dense(), out.to_dense())
            self.assertEqual(actual.to_dense(), expected)

        for index_dtype in [torch.int32, torch.int64]:
            for (m, n, k), noncontiguous in zip(itertools.product([1, 5], repeat=3), [True, False]):
                nnz = random.randint(0, m * n)
                c = self.genSparseCSRTensor((m, n), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a = make_tensor((m, k), dtype=dtype, device=device, noncontiguous=noncontiguous)
                b = make_tensor((k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_a, op_b in itertools.product([True, False], repeat=2):
                    run_test(c, a, b, op_a, op_b)

    @skipCUDAIfRocm
    @onlyCUDA
    @skipCUDAIf(True, "Causes CUDA memory exception, see https://github.com/pytorch/pytorch/issues/72177")
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_sampled_addmm_zero_sized(self, device, dtype):
        def run_test(c, a, b):
            actual = torch.sparse.sampled_addmm(c, a, b)
            self.assertEqual(actual.shape, c.shape)

        for m, n, k in itertools.product([0, 5], repeat=3):
            c = torch.empty(m, n, dtype=dtype, device=device, layout=torch.sparse_csr)
            a = make_tensor((m, k), dtype=dtype, device=device)
            b = make_tensor((k, n), dtype=dtype, device=device)
            run_test(c, a, b)

    @skipCUDAIfRocm
    @onlyCUDA
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_sampled_addmm_errors(self, device, dtype):
        # test that the errors are the same for dense and sparse sampled versions
        # import re

        # shapes must be compatible for matrix multiplication
        a = make_tensor((2, 3), dtype=dtype, device=device)
        a_sparse = a.to_sparse_csr()
        with self.assertRaisesRegex(RuntimeError, r"cannot be multiplied"):
            torch.sparse.sampled_addmm(a_sparse, a, a)

        # mat1 must be a matrix
        with self.assertRaisesRegex(RuntimeError, r"Expected mat1 to be a matrix"):
            torch.sparse.sampled_addmm(a_sparse, a.unsqueeze(0), a)

        # mat2 must be a matrix
        with self.assertRaisesRegex(RuntimeError, r"Expected mat2 to be a matrix"):
            torch.sparse.sampled_addmm(a_sparse, a, a.unsqueeze(0))

        a = make_tensor((2, 2), dtype=dtype, device=device)
        b = make_tensor((3, 3), dtype=dtype, device=device)
        b_sparse = b.to_sparse_csr()
        with self.assertRaisesRegex(RuntimeError, r"self dim 0 must match mat1 dim 0"):
            torch.sparse.sampled_addmm(b_sparse, a, a)

        b = make_tensor((2, 3), dtype=dtype, device=device)
        b_sparse = b.to_sparse_csr()
        with self.assertRaisesRegex(RuntimeError, r"self dim 1 must match mat2 dim 1"):
            torch.sparse.sampled_addmm(b_sparse, a, a)

        a = make_tensor((2, 2), dtype=dtype, device=device)
        a_sparse = a.to_sparse_csr()
        with self.assertRaisesRegex(RuntimeError, r"Expected mat1 to have strided layout"):
            torch.sparse.sampled_addmm(a_sparse, a_sparse, a_sparse)

        with self.assertRaisesRegex(RuntimeError, r"Expected mat2 to have strided layout"):
            torch.sparse.sampled_addmm(a_sparse, a, a_sparse)

    @dtypes(*get_all_dtypes())
    def test_coo_csr_conversion(self, device, dtype):
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            coo_sparse = dense.to_sparse()
            csr_sparse = coo_sparse.to_sparse_csr()

            self.assertEqual(csr_sparse.to_dense(), dense)

    @ops(_sparse_csr_ops)
    def test_sparse_csr_consistency(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)

        # Fail early to prevent silent success with this test
        ndims_equals_2d = (s.input.ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples.")

        for sample in samples:
            assert torch.is_tensor(sample.input)
            # Sparse CSR only supports 2D tensors as inputs
            if sample.input.ndim != 2:
                continue

            expected = op(sample.input)
            assert torch.is_tensor(expected)
            output = op(sample.input.to_sparse_csr())
            assert torch.is_tensor(output)

            self.assertEqual(output.to_dense(), expected)

    # Currently, there is no rule in PyTorch for filling zeros in the outputs
    #   from operations on Sparse CSR tensors. Hence only those operators are supported
    #   which have 0->0 correspondence, example: sin(0) = 0, tan(0) = 0 but
    #   cos(0) = 1 (and hence it's not supported).
    # Note: here, we do this test only for unary operators
    @ops(sparse_csr_unary_ufuncs)
    def test_zero_to_zero_correspondence_unary(self, device, dtype, op):
        zero = torch.zeros((1, 2), dtype=dtype, device=device)
        tensor_explicit_zeros = torch.sparse_csr_tensor([0, 1], [1], [0], dtype=dtype, device=device)

        output_zero = op(zero)
        expected_zero = zero.to(output_zero.dtype)

        output_explicit_zeros = op(tensor_explicit_zeros).to_dense()
        expected_explicit_zeros = tensor_explicit_zeros.to_dense().to(output_explicit_zeros.dtype)

        for (output, expected) in [
                (output_zero, expected_zero),
                (output_explicit_zeros, expected_explicit_zeros)
        ]:
            self.assertEqual(output, expected, f"This operator ({op.name}) should not be supported for "
                             "Sparse CSR as it breaks 0->0 correspondence.")

        for inp in [zero.to_sparse_csr(), tensor_explicit_zeros]:
            self.assertEqual(op(inp).values().numel(), inp.values().numel(),
                             f"{op.name} fails to preserve sparsity pattern.")

    @ops(sparse_csr_unary_ufuncs)
    def test_sparse_csr_unary_out(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)

        if not op.supports_out:
            self.skipTest("Skipped! Out not supported")

        for sample in samples:
            assert torch.is_tensor(sample.input)
            # Sparse CSR only supports 2D tensors as inputs
            # Fail early to prevent silent success with this test
            if sample.input.ndim != 2:
                raise ValueError("Expected 2D tensor but got tensor with dimension: {sample.input.ndim}.")

            sample.input = sample.input.to_sparse_csr()
            expect = op(sample.input, *sample.args, **sample.kwargs)

            out = self.genSparseCSRTensor(sample.input.size(), sample.input._nnz(),
                                          device=sample.input.device, dtype=expect.dtype,
                                          index_dtype=sample.input.crow_indices().dtype)
            op(sample.input, *sample.args, **sample.kwargs, out=out)

            self.assertEqual(out.values(), expect.values())
            self.assertEqual(out.crow_indices(), expect.crow_indices())
            self.assertEqual(out.col_indices(), expect.col_indices())
            self.assertEqual(out._nnz(), expect._nnz())

    @ops(sparse_csr_unary_ufuncs)
    def test_sparse_csr_unary_inplace(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)

        if op.inplace_variant is None:
            self.skipTest("Skipped! Inplace variant not supported!")

        for sample in samples:
            assert torch.is_tensor(sample.input)
            # Sparse CSR only supports 2D tensors as inputs
            # Fail early to prevent silent success with this test
            if sample.input.ndim != 2:
                raise ValueError("Expected 2D tensor but got tensor with dimension: {sample.input.ndim}.")

            sample.input = sample.input.to_sparse_csr()
            expect = op(sample.input, *sample.args, **sample.kwargs)

            if not torch.can_cast(expect.dtype, dtype):
                with self.assertRaisesRegex(RuntimeError, "result type"):
                    op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
                continue

            if sample.input.is_complex() and op.name == "abs":
                with self.assertRaisesRegex(RuntimeError, "not supported"):
                    op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
                continue

            actual = op.inplace_variant(sample.input, *sample.args, **sample.kwargs)

            self.assertIs(actual, sample.input)
            self.assertEqual(actual.values(), expect.values())
            self.assertEqual(actual.crow_indices(), expect.crow_indices())
            self.assertEqual(actual.col_indices(), expect.col_indices())
            self.assertEqual(actual._nnz(), expect._nnz())

    @dtypes(*get_all_dtypes(include_bool=False, include_half=False, include_bfloat16=False))
    def test_direct_coo_csr_conversion(self, device, dtype):
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            coo_sparse = dense.to_sparse_coo()

            self.assertEqual(coo_sparse.to_sparse_csr().to_sparse_coo(), coo_sparse)

    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_transpose(self, device, dtype):

        def run_test(shape, nnz, index_type, dim0, dim1):
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            t = a.transpose(dim0, dim1)

            self.assertEqual(t.to_dense(), a.to_dense().transpose(dim0, dim1))

        for shape, index_dtype, (dim0, dim1) in itertools.product(
                [(10, 5), (10, 10)],
                [torch.int32, torch.int64],
                [(0, 0), (0, 1)]):
            run_test(shape, 0, index_dtype, dim0, dim1)
            run_test(shape, max(shape), index_dtype, dim0, dim1)
            run_test(shape, shape[0] * shape[1], index_dtype, dim0, dim1)

    # TODO: This is a stopgap for a rigorous extension of our autograd tests
    # to test the functionality of detach
    @skipMeta
    @dtypes(*get_all_dtypes())
    def test_exercise_detach(self, device, dtype):
        shape = (3, 3)
        nnz = 4
        for index_dtype in [torch.int32, torch.int64]:
            inp = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            detached_inp = inp.detach()
            self.assertEqual(inp.values(), detached_inp.values())
            self.assertEqual(inp.crow_indices(), detached_inp.crow_indices())
            self.assertEqual(inp.col_indices(), detached_inp.col_indices())



# e.g., TestSparseCSRCPU and TestSparseCSRCUDA
instantiate_device_type_tests(TestSparseCSR, globals())

if __name__ == '__main__':
    run_tests()
