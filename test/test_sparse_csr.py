# Owner(s): ["module: sparse"]

import torch
import random
import itertools
import unittest
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC
from torch.testing._internal.common_utils import \
    (TEST_WITH_ROCM, TEST_SCIPY, TEST_MKL, IS_WINDOWS, TestCase, run_tests, load_tests, coalescedonoff, parametrize,
     subtest)
from torch.testing._internal.common_device_type import \
    (ops, instantiate_device_type_tests, dtypes, OpDTypes, dtypesIfCUDA, onlyCPU, onlyCUDA, skipCUDAIfNoCusparseGeneric,
     precisionOverride, skipMeta, skipCUDAIf, skipCUDAIfRocm, skipCPUIfNoMklSparse)
from torch.testing._internal.common_methods_invocations import \
    (op_db, sparse_csr_unary_ufuncs, ReductionOpInfo)
from torch.testing._internal.common_cuda import _get_torch_cuda_version, CUDA11OrLater
from torch.testing._internal.common_dtype import (
    floating_types, all_types_and_complex_and, floating_and_complex_types, floating_types_and,
    all_types_and_complex, floating_and_complex_types_and
)
from test_sparse import CUSPARSE_SPMM_COMPLEX128_SUPPORTED

if TEST_SCIPY:
    import scipy.sparse as sp

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

no_mkl_sparse = IS_WINDOWS or not TEST_MKL

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
_sparse_compressed_ops = list(filter(lambda op: (op.supports_sparse_csr or op.supports_sparse_csc
                                                 or op.supports_sparse_bsr or op.supports_sparse_bsc), op_db))
binary_functions_with_dense_output = ['mm', 'mv', ]
binary_ops_with_dense_output = list(filter(lambda op: op.name in binary_functions_with_dense_output, op_db))

UNARY_EWISE_CSR_ALLOW_AUTOGRAD = [
    'abs',
    'conj_physical',
    'neg',
]

# This should be just an import from test_linalg instead of code duplication
# but https://github.com/pytorch/pytorch/pull/63511#discussion_r733989701
def _test_addmm_addmv(
    test_case,
    f,
    t,
    m,
    v,
    *,
    alpha=None,
    beta=None,
    transpose_out=False,
    layout=torch.strided,
    mode=None
):
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

    if mode == "all_sparse":
        res1 = f(*map(convert_layout, (t, m, v)), alpha=alpha, beta=beta)
        res1 = res1.to_dense()
    elif mode == "dense_result":
        res1 = f(t, convert_layout(m), convert_layout(v), alpha=alpha, beta=beta)
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


def all_sparse_compressed_layouts(test_name='layout'):
    return parametrize(test_name, [
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC'),
        subtest(torch.sparse_bsr, name='SparseBSR'),
        subtest(torch.sparse_bsc, name='SparseBSC')])


def sparse_compressed_nonblock_layouts(test_name='layout'):
    return parametrize(test_name, [
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC')])


sparse_compressed_indices_methods = {
    torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
    torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
    torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
    torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
}


class TestSparseCompressed(TestCase):
    """Testing sparse compressed (CSR, CSC, BSR, BSC) tensor generic features.
    """

    def genTensor(self, size, nnz, *, layout, device=None, dtype=torch.float, index_dtype=torch.int64):
        if device is None:
            device = self.device_type
        return self.genSparseCompressedTensor(size, nnz, device=device, dtype=dtype, index_dtype=index_dtype, layout=layout)

    def _generate_small_inputs(self, layout, device, dtype, index_dtype):
        """Generator of inputs to sparse compressed tensor factory functions.

        The input is defined as a 4-tuple:
          compressed_indices, plain_indices, values, expected_size_from_shape_inference
        """
        from operator import mul
        from functools import reduce
        if layout in {torch.sparse_csr, torch.sparse_csc}:
            yield (torch.tensor([0, 2, 4], device=device, dtype=index_dtype),
                   torch.tensor([0, 1, 0, 1], device=device, dtype=index_dtype),
                   torch.tensor([1, 2, 3, 4], device=device, dtype=dtype),
                   (2, 2))
            yield (torch.tensor([0, ], device=device, dtype=index_dtype),
                   torch.tensor([], device=device, dtype=index_dtype),
                   torch.tensor([], device=device, dtype=dtype),
                   (0, 0))
            for batch_shape in [(2,), (2, 3)]:
                prod = reduce(mul, batch_shape, 1)
                yield (torch.tensor([0, 2, 4], device=device, dtype=index_dtype).repeat(prod, 1).reshape(*batch_shape, -1),
                       torch.tensor([0, 1, 0, 1], device=device, dtype=index_dtype).repeat(prod, 1).reshape(*batch_shape, -1),
                       torch.tensor([1, 2, 3, 4], device=device, dtype=dtype).repeat(prod, 1).reshape(*batch_shape, -1),
                       (*batch_shape, 2, 2))
        else:
            assert layout in {torch.sparse_bsr, torch.sparse_bsc}
            yield (torch.tensor([0, 2, 4], device=device, dtype=index_dtype),
                   torch.tensor([0, 1, 0, 1], device=device, dtype=index_dtype),
                   torch.tensor([[[1, 11]], [[2, 22]], [[3, 33]], [[4, 44]]], device=device, dtype=dtype),
                   (2, 4))
            yield (torch.tensor([0, ], device=device, dtype=index_dtype),
                   torch.tensor([], device=device, dtype=index_dtype),
                   torch.tensor([], device=device, dtype=dtype).reshape(1, 0, 0),
                   (0, 0))
            for batch_shape in [(2,), (2, 3)]:
                prod = reduce(mul, batch_shape, 1)
                yield (torch.tensor([0, 2, 4], device=device, dtype=index_dtype).repeat(prod, 1).reshape(*batch_shape, -1),
                       torch.tensor([0, 1, 0, 1], device=device, dtype=index_dtype).repeat(prod, 1).reshape(*batch_shape, -1),
                       torch.tensor([[[1, 11]], [[2, 22]], [[3, 33]], [[4, 44]]],
                                    device=device, dtype=dtype).repeat(prod, 1, 1).reshape(*batch_shape, 4, 1, 2),
                       (*batch_shape, 2, 4))

    @all_sparse_compressed_layouts()
    @onlyCPU
    def test_layout(self, layout):
        self.assertIn(str(layout), {'torch.sparse_csr', 'torch.sparse_csc', 'torch.sparse_bsr', 'torch.sparse_bsc'})
        self.assertEqual(type(layout), torch.layout)

    @parametrize('shape_and_device_inference', [subtest(False, name='_'), subtest(False, name='shape_and_device_inference')])
    @parametrize('use_factory_function', [subtest(False, name='_'), subtest(True, name='factory')])
    @parametrize('input_kind', [subtest('tensor', name='from_tensor'), subtest('list', name='from_list')])
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_compressed_constructor(self, layout, device, dtype,
                                           use_factory_function, shape_and_device_inference, input_kind):
        factory_function = {
            torch.sparse_csr: torch.sparse_csr_tensor,
            torch.sparse_csc: torch.sparse_csc_tensor,
            torch.sparse_bsr: torch.sparse_bsr_tensor,
            torch.sparse_bsc: torch.sparse_bsc_tensor,
        }[layout]
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        for index_dtype in [torch.int32, torch.int64]:
            for compressed_indices, plain_indices, values, size in self._generate_small_inputs(layout, device, dtype, index_dtype):
                if input_kind == 'list':
                    if size == (0, 0):
                        # for this degenerate case, plain_indices must
                        # remain a tensor because
                        # tensor(plain_indices) results a float dtype
                        # when plain_indices is an empty list
                        if index_dtype == torch.int32:
                            # skip testing int32 case because
                            # tensor(compressed_indices) results a
                            # int64 dtype when compressed_indices is
                            # [0] (a list of single int zero).
                            continue
                    else:
                        plain_indices = plain_indices.tolist()
                    compressed_indices = compressed_indices.tolist()
                    values = values.tolist()
                    if size == (0, 0) and layout in {torch.sparse_bsr, torch.sparse_bsc}:
                        # in the block sparse case, values of type list needs to represent a 3-D tensor
                        values = [[[]]]
                if use_factory_function:
                    if shape_and_device_inference:
                        sparse = factory_function(compressed_indices, plain_indices, values)
                    else:
                        sparse = factory_function(compressed_indices, plain_indices, values, size,
                                                  dtype=dtype, device=device)
                else:
                    if shape_and_device_inference:
                        sparse = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=layout)
                    else:
                        sparse = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size,
                                                                dtype=dtype, layout=layout, device=device)
                self.assertEqual(layout, sparse.layout)
                self.assertEqual(size, sparse.shape)
                self.assertEqual(compressed_indices, compressed_indices_mth(sparse))
                self.assertEqual(plain_indices, plain_indices_mth(sparse))
                self.assertEqual(values, sparse.values())

    @skipMeta
    @sparse_compressed_nonblock_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half))
    def test_empty(self, layout, device, dtype):
        ns = [5, 2, 0]
        batch_shapes = [(), (2,), (2, 3)]
        compressed_dim = {
            torch.sparse_csr: -2,
            torch.sparse_csc: -1,
        }[layout]
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        for m, n, b in itertools.product(ns, ns, batch_shapes):
            shape = (*b, m, n)
            result = torch.empty(shape, dtype=dtype, device=device, layout=layout)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, torch.device(device))
            self.assertEqual(result.layout, layout)
            self.assertEqual(compressed_indices_mth(result).shape, (*b, shape[compressed_dim] + 1,))
            self.assertEqual(plain_indices_mth(result).shape, (*b, 0,))
            self.assertEqual(result.values().shape, (*b, 0,))
            self.assertEqual(result._nnz(), 0)
            self.assertEqual(compressed_indices_mth(result).device, torch.device(device))
            self.assertEqual(plain_indices_mth(result).device, torch.device(device))
            self.assertEqual(result.values().device, torch.device(device))
            self.assertEqual(compressed_indices_mth(result).dtype, torch.int64)
            self.assertEqual(plain_indices_mth(result).dtype, torch.int64)
            self.assertEqual(result.values().dtype, dtype)

    @skipMeta
    @sparse_compressed_nonblock_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_empty_errors(self, layout, device, dtype):
        with self.assertRaisesRegex(RuntimeError,
                                    "torch.empty: Only batched sparse compressed \\(non-block\\) tensors are supported"
                                    ", but got size"):
            torch.empty((5,), dtype=dtype, device=device, layout=layout)

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_clone(self, layout, device, dtype):
        for compressed_indices, plain_indices, values, size in self._generate_small_inputs(
                layout, device, dtype, index_dtype=torch.int32):
            sparse = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size,
                                                    dtype=dtype, layout=layout, device=device)
            cloned_sparse = sparse.clone()
            self.assertEqual(sparse, cloned_sparse)

    @all_sparse_compressed_layouts()
    def test_print(self, layout, device):
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        printed = []
        for index_dtype in [torch.int32, torch.int64]:
            for dtype in [torch.float32, torch.float64]:
                for compressed_indices, plain_indices, values, size in self._generate_small_inputs(
                        layout, device, dtype, index_dtype):
                    batch_shape = tuple(size[:-2])
                    block_shape = tuple(values.shape[-2:]) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
                    blocksize0, blocksize1 = block_shape if layout in {torch.sparse_bsr, torch.sparse_bsc} else (1, 1)
                    if size not in [(2 * blocksize0, 2 * blocksize1), (0, 0),
                                    (2, 3, 2 * blocksize0, 2 * blocksize1), (2, 2 * blocksize0, 2 * blocksize1)]:
                        # Skip inputs that are not in the list of
                        # expected sizes to ensure the stability of
                        # test_print in the case
                        # _generate_small_inputs is extended with new
                        # inputs
                        continue
                    if block_shape not in [(), (0, 0), (1, 2)]:
                        # Skip inputs that are not in the list of
                        # expected block sizes to ensure test_print
                        # stability.
                        continue
                    printed.append("########## {}/{}/batch_shape={}/block_shape={} ##########".format(
                        dtype, index_dtype, batch_shape, block_shape))
                    x = torch.sparse_compressed_tensor(compressed_indices,
                                                       plain_indices,
                                                       values, dtype=dtype, layout=layout, device=device)
                    printed.append("# sparse tensor")
                    printed.append(str(x))
                    printed.append(f"# _{compressed_indices_mth.__name__}")
                    printed.append(str(compressed_indices_mth(x)))
                    printed.append(f"# _{plain_indices_mth.__name__}")
                    printed.append(str(plain_indices_mth(x)))
                    printed.append("# _values")
                    printed.append(str(x.values()))
                    printed.append('')
                printed.append('')
        orig_maxDiff = self.maxDiff
        self.maxDiff = None
        try:
            self.assertExpected('\n'.join(printed))
            self.maxDiff = orig_maxDiff
        except Exception:
            self.maxDiff = orig_maxDiff
            raise

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_copy(self, layout, device, dtype):

        def run_test(shape, blocksize, nnz, index_type):
            a = self.genSparseCompressedTensor(shape, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            b = self.genSparseCompressedTensor(shape, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)

            a.copy_(b)

            self.assertEqual(a, b)

        ns = [(9, 3), (2, 1), (0, 0)]  # (number of dimensions, the corresponding block size)
        batch_shapes = [(), (2,), (2, 3)]
        for ((m, bm), (n, bn), b), index_dtype in zip(itertools.product(ns, ns, batch_shapes), [torch.int32, torch.int64]):
            blocksize = (bm, bn) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
            run_test((*b, m, n), blocksize, 0, index_dtype)
            run_test((*b, m, n), blocksize, m * n, index_dtype)

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_copy_errors(self, layout, device, dtype):
        blocksize = (2, 3) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        nnz = 6 if layout in {torch.sparse_bsr, torch.sparse_bsc} else 1
        shape1 = (2 * 6, 3 * 6) if layout in {torch.sparse_bsr, torch.sparse_bsc} else (2, 3)
        for index_dtype in [torch.int32, torch.int64]:
            a = self.genSparseCompressedTensor(shape1, 0, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)

            with self.assertRaisesRegex(RuntimeError,
                                        "copy of sparse compressed tensors having different layouts is not supported."):
                a.copy_(torch.empty(a.shape, dtype=dtype, device=device))

            b = self.genSparseCompressedTensor(shape1, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            assert a._nnz() != b._nnz(), (a._nnz(), b._nnz())
            with self.assertRaisesRegex(RuntimeError,
                                        "only sparse compressed tensors with the same number of specified elements are supported."):
                a.copy_(b)

            shape2 = tuple(reversed(shape1))
            c = self.genSparseCompressedTensor(shape2, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "expected shapes of self and src to match along dimension"):
                b.copy_(c)

            if blocksize:
                blocksize1 = tuple(reversed(blocksize))
                d = self.genSparseCompressedTensor(shape1, nnz, dtype=dtype, layout=layout, device=device,
                                                   index_dtype=index_dtype, blocksize=blocksize1)
                with self.assertRaisesRegex(RuntimeError,
                                            "copy of sparse compressed tensors having different block sizes is not supported"):
                    b.copy_(d)

    def _smallest_divisor(self, n):
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return i
        return n

    @all_sparse_compressed_layouts()
    @ops(_sparse_compressed_ops)
    def test_consistency(self, layout, device, dtype, op):
        # TODO: Normaly, we should use DecorateInfo instead of
        # skipTest but this requires implemening OpInfo support for
        # layout as a test parameter (similar to device and dtype).
        if not (layout == torch.sparse_csr and op.supports_sparse_csr
                or layout == torch.sparse_csc and op.supports_sparse_csc
                or layout == torch.sparse_bsr and op.supports_sparse_bsr
                or layout == torch.sparse_bsc and op.supports_sparse_bsc):
            self.skipTest(f"{op.name} does not support input with {layout} layout")

        require_mask = isinstance(op, ReductionOpInfo) and '_masked.' in op.name
        if require_mask and layout in {torch.sparse_bsr, torch.sparse_bsc}:
            self.skipTest(f"{op.name} does not support input with {layout} layout")

        if layout is torch.sparse_bsc:
            self.skipTest(f"test requires conversion from Strided layout to {layout} layout")

        samples = list(op.sample_inputs(device, dtype))

        # Fail early to prevent silent success with this test
        ndims_equals_2d = (s.input.ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples.")

        count = 0
        for sample in samples:
            assert torch.is_tensor(sample.input)
            # Sparse CSR/CSC only supports 2D tensors as inputs
            if sample.input.ndim != 2:
                continue
            if isinstance(op, ReductionOpInfo):
                # Reductions on sparse compressed require keepdim=True
                if not sample.kwargs.get('keepdim'):
                    continue
                # Reductions on sparse compressed tensors require explicit mask
                if require_mask and sample.kwargs.get('mask') is None:
                    continue
            expected = op(sample.input, **sample.kwargs)
            assert torch.is_tensor(expected)
            # Use smallest non-trivial blocksize for the given input shape:
            blocksize = tuple(map(self._smallest_divisor, sample.input.shape[-2:]))
            if layout is torch.sparse_bsr:
                sparse = sample.input.to_sparse_bsr(blocksize)
            elif layout is torch.sparse_bsc:
                sparse = sample.input.to_sparse_bsc(blocksize)
            elif layout is torch.sparse_csr:
                sparse = sample.input.to_sparse_csr()
            elif layout is torch.sparse_csc:
                sparse = sample.input.to_sparse_csc()
            else:
                assert 0, layout

            assert torch.is_tensor(sparse)
            output = op(sparse, **sample.kwargs)
            assert torch.is_tensor(output)
            strided_output = output.to_dense()
            if require_mask:
                expected *= torch._masked._output_mask(op.op, sample.input, **sample.kwargs)
            self.assertEqual(strided_output, expected)
            count += 1

        # Better fail late to prevent silent success with this test
        if not count:
            raise ValueError("Expected at least one sample with keepdim and/or explicit mask for reductions.")


class TestSparseCSR(TestCase):

    def test_csr_stride(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have strides"):
            a.stride()

        with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have strides"):
            a.stride(-1)

    def test_csr_storage(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "Cannot access storage of SparseCsrTensorImpl"):
            a.storage()

    def test_csr_is_contiguous(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "Tensors of type SparseCsrTensorImpl do not have is_contiguous"):
            a.is_contiguous()

    def test_csr_double_to_sparse_csr(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)
        a.to_sparse_csr().to_sparse_csr()

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_sparse_csr_select(self, device, dtype):
        batch_shape = (2, 3)
        crow_indices = torch.tensor([0, 2, 4], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        col_indices = torch.tensor([0, 1, 0, 1], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        values = torch.tensor([1, 2, 3, 4], device=device, dtype=dtype).repeat(6, 1).reshape(*batch_shape, -1)
        sparse = torch.sparse_csr_tensor(crow_indices,
                                         col_indices,
                                         values,
                                         size=(*batch_shape, 2, 10),
                                         dtype=dtype,
                                         device=device)

        # select from batch dimensions
        sparse_selected12 = sparse.select(1, 2)
        expected_sparse_selected12 = torch.sparse_csr_tensor(crow_indices.select(1, 2).contiguous(),
                                                             col_indices.select(1, 2).contiguous(),
                                                             values.select(1, 2).contiguous(),
                                                             size=(2, 2, 10),
                                                             dtype=dtype,
                                                             device=device)
        self.assertEqual(expected_sparse_selected12, sparse_selected12)

        # select from rows or columns
        sparse_non_batched = sparse[0, 0]
        for selects_args in [(0, 0), (1, 1)]:
            sparse_selected = sparse_non_batched.select(*selects_args)
            dense_selected = sparse_non_batched.to_dense().select(*selects_args)
            self.assertEqual(dense_selected, sparse_selected)

        # index a single element
        self.assertEqual(sparse[0, 0, 0, 0], sparse.to_dense()[0, 0, 0, 0])

        # selecting from rows or columns for batched CSR is not yet implemented
        with self.assertRaisesRegex(RuntimeError, "selecting rows or columns is not implemented for batched"):
            sparse.select(-2, 0)

        with self.assertRaisesRegex(RuntimeError, "selecting rows or columns is not implemented for batched"):
            sparse.select(-1, 0)

        # assigning to sparse trhough indexing is disabled
        with self.assertRaisesRegex(TypeError, "Cannot assign to a sparse tensor"):
            sparse[0, 0, 0, 0] = 99.0

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_resize(self, device, dtype):
        batch_shapes = [(), (2,), (2, 3)]
        for index_dtype, b in zip([torch.int32, torch.int64], batch_shapes):
            shape = (*b, 2, 3)
            nnz = 6
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            new_shape = (*b, 4, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to larger shape doesn't add specified elements
            self.assertEqual(a._nnz(), nnz)

            new_shape = (*b, 1, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to smaller shape trims specified elements
            self.assertEqual(a._nnz(), 5)

            # trim batched dimensions
            a.resize_(new_shape[-2], new_shape[-1])
            self.assertEqual(a.shape, (new_shape[-2], new_shape[-1]))
            self.assertEqual(a._nnz(), 5)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_resize_errors(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            shape = (2, 3)
            nnz = 6
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            with self.assertRaisesRegex(RuntimeError, "torch.resize_: Only batched sparse CSR matrices are supported"):
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

        with self.assertRaisesRegex(RuntimeError, "crow_indices and col_indices must be an int32 or int64 type"):
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
        crow_indices = torch.tensor([0, 2, 4], device=device)
        col_indices = torch.tensor([0, 1, 0, 1], device=device)
        values = torch.tensor([1, 2, 3, 4], device=device)
        size = (2, 10)
        torch.sparse_csr_tensor(crow_indices, col_indices, values, size, device=device)

        with self.assertRaisesRegex(RuntimeError, r"size of a batched CSR tensor must have length >= 2, but got: 1"):
            torch.sparse_csr_tensor(crow_indices, col_indices, values,
                                    size=(2,),
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"crow_indices must have dim >= 1 but got crow_indices\.dim\(\)\ = 0"):
            torch.sparse_csr_tensor(torch.zeros((), device=device, dtype=torch.int64),
                                    col_indices,
                                    values,
                                    size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"col_indices must have dim >= 1 but got col_indices\.dim\(\)\ = 0"):
            torch.sparse_csr_tensor(crow_indices,
                                    torch.zeros((), device=device, dtype=torch.int64),
                                    values,
                                    size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"values must have dim >= 1 but got values\.dim\(\)\ = 0"):
            torch.sparse_csr_tensor(crow_indices,
                                    col_indices,
                                    torch.zeros((), device=device, dtype=torch.int64),
                                    size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    r"crow_indices\.size\(-1\) must be equal to size\[-2\] \+ 1 \(that is 2\), but got: 3"):
            torch.sparse_csr_tensor(crow_indices, col_indices, values, (1, 1),
                                    device=device)


        with self.assertRaisesRegex(RuntimeError,
                                    r"number of dimensions of crow_indices and col_indices must be the same"):
            torch.sparse_csr_tensor(crow_indices, col_indices.repeat(2, 1), values, size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    r"non-zero dense dimensions \(=1\) is not supported"):
            torch.sparse_csr_tensor(crow_indices, col_indices, values.repeat(2, 1), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    r"number of dimensions of indices must be one less"):
            torch.sparse_csr_tensor(crow_indices.repeat(2, 1), col_indices.repeat(2, 1), values.repeat(2, 1), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    r"all batch dimensions of the provided size \(\[2\]\), indices \(\[2\], \[3\]\),"
                                    r" and values \(\[4\]\) must be the same"):
            torch.sparse_csr_tensor(crow_indices.repeat(2, 1), col_indices.repeat(3, 1), values.repeat(4, 1), (2, 2, 10),
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
                                    r" the condition crow_indices\[i - 1\] <\= crow_indices\[i\] fails"):
            torch.sparse_csr_tensor(torch.tensor([0, 5, 4]), torch.tensor(col_indices), torch.tensor(values), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"col_indices\.min\(\) should be greater or equal to zero"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor([0, -1, 0, 1]), torch.tensor(values), size,
                                    device=device)

        with self.assertRaisesRegex(RuntimeError, r"size\[-1\] should be greater than col_indices\.max\(\)"):
            torch.sparse_csr_tensor(torch.tensor(crow_indices), torch.tensor([0, 11, 0, 1]), torch.tensor(values), size,
                                    device=device)

    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
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

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
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

    def _test_sparse_compressed_to_dense(self, device, dtype, layout):
        compressed_format_str = str(layout)[-3:]

        def to_compressed(t):
            return getattr(t, f"to_sparse_{compressed_format_str}")()

        def compressed_constructor(*input, **kwargs):
            constructor = getattr(torch, f"sparse_{compressed_format_str}_tensor")
            return constructor(*input, **kwargs)

        def get_dense_shape(shape, batch_ndim):
            if layout is torch.sparse_csc:
                compressed_dims_slice = slice(batch_ndim + 1, batch_ndim - 1, -1)
            else:
                compressed_dims_slice = slice(batch_ndim, batch_ndim + 2)
            return shape[:batch_ndim] + shape[compressed_dims_slice] + shape[batch_ndim + 2:]

        def transpose(t, batch_ndim):
            if layout is torch.sparse_csc:
                return t.transpose(batch_ndim, batch_ndim + 1)
            return t

        mn = [5, 2, 0]
        for (m, n) in itertools.product(mn, mn):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            sparse = to_compressed(dense)
            self.assertEqual(sparse.to_dense(), dense)

        batch_shape = (2, 3)
        compressed_indices = torch.tensor([0, 3, 5], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        plain_indices = torch.tensor([0, 1, 2, 0, 1], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        values = torch.tensor([1, 2, 1, 3, 4], device=device, dtype=dtype).repeat(6, 1).reshape(*batch_shape, -1)
        sparse = compressed_constructor(compressed_indices, plain_indices, values, dtype=dtype, device=device)
        dense_shape = get_dense_shape(sparse.shape, len(batch_shape))
        dense = torch.tensor([[1, 2, 1], [3, 4, 0]], dtype=dtype, device=device).repeat(6, 1).reshape(dense_shape)
        self.assertEqual(sparse.to_dense(), transpose(dense, len(batch_shape)))

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_csr_to_dense(self, device, dtype):
        self._test_sparse_compressed_to_dense(device, dtype, torch.sparse_csr)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_csc_to_dense(self, device, dtype):
        self._test_sparse_compressed_to_dense(device, dtype, torch.sparse_csc)

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

    @parametrize("blocksize", [2, 4])
    @dtypes((torch.double, torch.int32), (torch.double, torch.int64))
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @skipMeta
    def test_csr_to_block_csr(self, device, dtypes, blocksize):
        for shape in [(24, 24), (12, 24)]:
            dtype, index_dtype = dtypes
            m, k = shape
            nnz = random.randint(0, m * k)
            t = self.genSparseCSRTensor((m * blocksize, k * blocksize), nnz, dtype=dtype,
                                        device=device, index_dtype=index_dtype)
            st = sp.csr_matrix((t.values().cpu(), t.col_indices().cpu(), t.crow_indices().cpu()), shape=tuple(t.size()))
            block_t = t.to_sparse_bsr((blocksize, blocksize))
            self.assertEqual(block_t.values().dim(), 3)
            self.assertTrue(block_t.layout == torch.sparse_bsr)
            block_st = st.tobsr(blocksize=(blocksize, blocksize))
            self.assertEqual(block_t.values().cpu(), block_st.data)
            self.assertEqual(block_t.col_indices().cpu(), torch.tensor(block_st.indices).to(index_dtype))
            self.assertEqual(block_t.crow_indices().cpu(), torch.tensor(block_st.indptr).to(index_dtype))

    @dtypes(torch.double)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_csr_to_block_csr_errors(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            nnz = 15
            t = self.genSparseCSRTensor((16, 16), nnz, dtype=dtype,
                                        device=device, index_dtype=index_dtype)
            with self.assertRaisesRegex(RuntimeError, "must be square."):
                block_t = t.to_sparse_bsr((2, 3))

            with self.assertRaisesRegex(RuntimeError, r"size \(16, 16\) with block size \(5, 5\)"):
                block_t = t.to_sparse_bsr((5, 5))

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
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
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater else [],
                  *[torch.bfloat16] if SM80OrLater else []))
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
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([2, 5], repeat=3), [1, 3], [True, False]):
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
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([2, 5], repeat=3), [1, 3], [True, False]):
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

    # TODO: block_size 1 is broken
    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-5, torch.complex128: 1e-5})
    def test_block_addmm(self, device, dtype, index_dtype, block_size, noncontiguous):
        for (m, n, k) in itertools.product([2, 5], repeat=3):
            nnz = random.randint(0, m * k)
            if not noncontiguous:
                a = self.genSparseCSRTensor((m * block_size, k * block_size), nnz,
                                            dtype=dtype, device=device, index_dtype=index_dtype)
                a = a.to_sparse_bsr((block_size, block_size))
            else:
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data   # Test column-major blocks
                a = torch._sparse_bsr_tensor_unsafe(a.crow_indices(), a.col_indices(),
                                                    a_data, (m * block_size, k * block_size))
            b = make_tensor((k * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
            c = make_tensor((m * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
            for op_b, op_out in itertools.product([True, False], repeat=2):
                self.run_test_block_addmm_addmv(torch.addmm, c, a, b, op_b, op_out, dtype=dtype, device=device)

    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_addmv(self, device, dtype, index_dtype, block_size, noncontiguous):
        # TODO: Explicitly disable block size 1 support
        # if (TEST_WITH_ROCM or not TEST_CUSPARSE_GENERIC) and block_size == 1:
        #     return
        for (m, k) in itertools.product([2, 5], repeat=2):
            nnz = random.randint(0, m * k)
            if not noncontiguous:
                a = self.genSparseCSRTensor((m * block_size, k * block_size), nnz,
                                            dtype=dtype, device=device, index_dtype=index_dtype)
                a = a.to_sparse_bsr((block_size, block_size))
            else:
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data   # Test column-major blocks
                a = torch._sparse_bsr_tensor_unsafe(a.crow_indices(), a.col_indices(),
                                                    a_data, (m * block_size, k * block_size))
            b = make_tensor((k * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            c = make_tensor((m * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            self.run_test_block_addmm_addmv(torch.addmv, c, a, b, dtype=dtype, device=device)

    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_triangular_solve(self, device, dtype, index_dtype, block_size, noncontiguous):
        def run_test(a, b, upper, transpose, unitriangular, op_out):
            if unitriangular and self.device_type == 'cpu':
                # TODO: When unitriangular=True results are not correct on CPU
                return

            if not upper and self.device_type == 'cpu':
                # TODO: When upper=False some generated inputs might crash on CPU
                return

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

            if expected_X.isnan().any():
                # TODO: zeros on the diagonal are not handled for CPU path
                # there's no way to query this info from MKL
                if self.device_type == 'cuda' and not TEST_WITH_ROCM:
                    self.assertTrue(actual_X.isnan().any() or actual_X.isinf().any())
                return

            self.assertEqual(actual_X, expected_X)

            out = torch.empty_like(b.mH if op_out and a.shape == b.shape else b)
            torch.triangular_solve(
                b, a,
                upper=upper, unitriangular=unitriangular, transpose=transpose, out=(out, actual_A_clone)
            )
            self.assertEqual(out, actual_X)
            self.assertEqual(out, expected_X)

        for (m, k) in itertools.product([2, 3], [1, 3]):
            nnz = random.randint(0, m * m)
            if not noncontiguous:
                a = self.genSparseCSRTensor((m * block_size, m * block_size), nnz,
                                            dtype=dtype, device=device, index_dtype=index_dtype)
                a = a.to_sparse_bsr((block_size, block_size))
            else:
                a = self.genSparseCSRTensor((m, m), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data  # Test column-major blocks
                a = torch._sparse_bsr_tensor_unsafe(a.crow_indices(), a.col_indices(),
                                                    a_data, (m * block_size, m * block_size))
            b = make_tensor((m * block_size, k), dtype=dtype, device=device, noncontiguous=noncontiguous)

            for (upper, unitriangular, transpose, op_out) in itertools.product([True, False], repeat=4):
                run_test(a, b, upper, unitriangular, transpose, op_out)

    @skipCPUIfNoMklSparse
    @unittest.skipIf(not CUDA11OrLater, "Only CUDA 11+ is supported")
    @dtypes(torch.double)
    def test_mm(self, device, dtype):
        def test_shape(di, dj, dk, nnz0=None, nnz1=None):
            for index_dtype in [torch.int32, torch.int64]:
                alpha = random.random()
                beta = random.random()

                def _test_addmm(t, x, y):
                    # TODO: addmm doesn't support strided result for sparse inputs.
                    # res = beta * t  + alpha * (x @ y)
                    res = torch.addmm(t, x, y, beta=beta, alpha=alpha)
                    expected = torch.addmm(t, x.to_dense(), y.to_dense(), beta=beta, alpha=alpha)
                    self.assertEqual(res, expected)

                    res = torch.addmm(t, x, y)
                    expected = torch.addmm(t, x.to_dense(), y.to_dense())
                    self.assertEqual(res, expected)

                def _test_mm(x, y):
                    res = torch.mm(x, y)
                    expected = torch.mm(x.to_dense(), y.to_dense())
                    if x.layout is torch.strided or y.layout is torch.strided:
                        self.assertEqual(res.layout, torch.strided)
                    else:
                        self.assertEqual(res.layout, torch.sparse_csr)
                    self.assertEqual(res.to_dense(), expected)

                def _test(t, x, y):
                    _test_addmm(t, x, y)
                    _test_mm(x, y)

                if nnz0 is None:
                    nnz0 = random.randint(di * dk // 2, di * dk)
                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = self.genSparseCSRTensor((di, dk), nnz0, device=device, dtype=dtype, index_dtype=index_dtype)
                y = torch.randn(dk, dj, dtype=dtype, device=device)
                _test(t, x, y)

                if nnz1 is None:
                    nnz1 = random.randint(dk * dj // 2, dk * dj)
                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = torch.randn(di, dk, dtype=dtype, device=device)
                y = self.genSparseCSRTensor((dk, dj), nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                _test(t, x, y)

                x_shape, y_shape = x.shape, y.shape

                gen_csr_csc = [self.genSparseCSRTensor, self.genSparseCSCTensor]

                # Test mm({CSR, CSC}, {CSR, CSC})
                for gen_x, gen_y in itertools.product(gen_csr_csc, gen_csr_csc):
                    x = gen_x(x_shape, nnz0, device=device, dtype=dtype, index_dtype=index_dtype)
                    y = gen_y(y_shape, nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                    _test_mm(x, y)

        for i in [2, 4]:
            for j in [2, 4, 7]:
                for k in [2, 3, 7]:
                    test_shape(i, j, k)
        test_shape(4, 4, 4, 0, 0)

    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater and TEST_CUSPARSE_GENERIC else [],
                  *[torch.bfloat16] if SM80OrLater and TEST_CUSPARSE_GENERIC else []))
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

    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater and TEST_CUSPARSE_GENERIC else [],
                  *[torch.bfloat16] if SM80OrLater and TEST_CUSPARSE_GENERIC else []))
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
    @dtypesIfCUDA(*floating_types_and(torch.complex64,
                                      *[torch.bfloat16] if SM80OrLater else [],
                                      *[torch.half] if SM53OrLater else [],
                                      *[torch.complex128] if CUSPARSE_SPMM_COMPLEX128_SUPPORTED else []))
    @skipCUDAIf(
        not _check_cusparse_spgemm_available(),
        "cuSparse Generic API SpGEMM is not available"
    )
    def test_addmm_all_sparse_csr(self, device, dtype):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, mode="all_sparse")

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, mode="all_sparse")

        # Test beta=0, M=nan
        M = torch.full((10, 25), float('nan'), device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, beta=0, layout=torch.sparse_csr, mode="all_sparse")

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            _test_addmm_addmv(self, torch.addmm, M, m1, m2, transpose_out=t4, layout=torch.sparse_csr, mode="all_sparse")

    @onlyCPU
    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    def test_addmm_dense_result(self, device, dtype):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, mode="dense_result")

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, mode="dense_result")

        # Test beta=0, M=nan
        M = torch.full((10, 25), float('nan'), device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, beta=0, layout=torch.sparse_csr, mode="dense_result")

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            _test_addmm_addmv(self, torch.addmm, M, m1, m2, transpose_out=t4, layout=torch.sparse_csr, mode="dense_result")

    @parametrize("k", [0, 1, 8])
    @parametrize("n", [0, 1, 10])
    @parametrize("m", [0, 1, 25])
    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_types_and(torch.complex64,
                                      *[torch.bfloat16] if SM80OrLater else [],
                                      *[torch.half] if SM53OrLater else [],
                                      *[torch.complex128] if CUSPARSE_SPMM_COMPLEX128_SUPPORTED else []))
    @skipCUDAIf(
        not _check_cusparse_spgemm_available(),
        "cuSparse Generic API SpGEMM is not available"
    )
    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    def test_addmm_sizes_all_sparse_csr(self, device, dtype, m, n, k):
        M = torch.randn(n, m, device=device).to(dtype)
        m1 = torch.randn(n, k, device=device).to(dtype)
        m2 = torch.randn(k, m, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, mode="all_sparse")

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
            # sparse.to_dense() uses torch.add internally so if torch.add is wrong,
            # the dense tensor will be wrong but this test would still pass
            # there's a separate test that checks for the correctness of the .to_dense() call
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

        ns = [2, 5]
        batch_shapes = [(), (2,), (2, 3)]
        for b, m, n in itertools.product(batch_shapes, ns, ns):
            _test_spadd_shape(0, (*b, m, n))
            _test_spadd_shape(m * n // 2, (*b, m, n))
            _test_spadd_shape(m * n, (*b, m, n))

    @dtypes(torch.float, torch.double)
    def test_mul(self, device, dtype):
        # TODO: This whole test should be migrated to OpInfos
        def _test_spadd_shape(fn, nnz, shape):
            x = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=torch.int32)
            y = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=torch.int32)

            # Forward comparison
            res_sparse_sparse = fn(y, x)
            res_dense_sparse = fn(y.to_dense(), x)
            res_sparse_dense = fn(y, x.to_dense())
            expected = fn(y.to_dense(), x.to_dense()).to_sparse_csr()
            self.assertEqual(res_sparse_sparse, expected)
            # TODO: While result of mul(dense, csr) is csr, it is not fully compressed.
            # That means it may contain materialized zeros, since the dense argument
            # is converted according to the sparsity pattern of csr. In the future
            # we might require the result to be fully compressed.
            self.assertEqual(res_dense_sparse.to_dense(), expected.to_dense())
            self.assertEqual(res_sparse_dense.to_dense(), expected.to_dense())

            # Grad comparison
            x = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=torch.int32)
            y = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=torch.int32)
            z = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=torch.int32)

            # csr * csr -> csr with csr, csr gradients
            x_a = x.clone().requires_grad_()
            y_a = y.clone().requires_grad_()

            fn(y_a, x_a).backward(z)

            x_dense_a = x.to_dense().requires_grad_()
            y_dense_a = y.to_dense().requires_grad_()

            fn(y_dense_a, x_dense_a).backward(z.to_dense())

            self.assertEqual(x_a.grad.layout, torch.sparse_csr)
            self.assertEqual(y_a.grad.layout, torch.sparse_csr)

            self.assertEqual(x_a.grad.to_dense(), x_dense_a.grad)
            self.assertEqual(y_a.grad.to_dense(), y_dense_a.grad)

            # TODO: Currently strided Tensors cannot have csr gradients
            # dense * csr -> csr with csr, dense gradients
            x_a = x.clone().requires_grad_()
            y_a = y.to_dense().clone().requires_grad_()
            err_msg = "Function MulBackward0 returned an invalid gradient at index 0 - expected layout Strided but got SparseCsr"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                fn(y_a, x_a).backward(z)

            # csr * dense -> csr with dense, csr gradients
            x_a = x.to_dense().clone().requires_grad_()
            y_a = y.clone().requires_grad_()
            err_msg = "Function MulBackward0 returned an invalid gradient at index 1 - expected layout Strided but got SparseCsr"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                fn(y_a, x_a).backward(z)

        _test_spadd_shape(torch.mul, 100, [100, 100])
        _test_spadd_shape(torch.mul, 0, [100, 100])
        _test_spadd_shape(torch.mul, 100, [100, 1])
        _test_spadd_shape(torch.mul, 100, [1, 100])

    @skipCPUIfNoMklSparse
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_sparse_add(self, device, dtype):
        def run_test(m, n, index_dtype):

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
                size=actual.shape
            )
            torch.sparse.sampled_addmm(c, a, b, alpha=alpha, beta=beta, out=out)

            spy_c = torch.sparse_csr_tensor(c.crow_indices(), c.col_indices(), torch.ones_like(c.values()), size=c.shape)
            expected = alpha * (a @ b) * spy_c.to_dense() + beta * c.to_dense()
            self.assertEqual(actual.to_dense(), out.to_dense())
            self.assertEqual(actual.to_dense(), expected)

        mnk = itertools.product([2, 5], repeat=3)
        batch_shapes = [(), (2,), (2, 3)] if self.device_type == 'cuda' else [(), ]
        tf = [True, False]
        for index_dtype in [torch.int32, torch.int64]:
            for (m, n, k), b, noncontiguous, bcast_c in itertools.product(mnk, batch_shapes, tf, tf):
                if bcast_c and len(b) == 0:
                    continue
                nnz = random.randint(0, m * n)
                c_batch = () if bcast_c else b
                c = self.genSparseCSRTensor((*c_batch, m, n), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a = make_tensor((*b, m, k), dtype=dtype, device=device, noncontiguous=noncontiguous)
                b = make_tensor((*b, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_a, op_b in itertools.product([True, False], repeat=2):
                    run_test(c, a, b, op_a, op_b)

    @skipCUDAIfRocm
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_sampled_addmm_autograd(self, device, dtype):
        from torch.testing._internal.common_methods_invocations import sample_inputs_sparse_sampled_addmm

        samples = list(sample_inputs_sparse_sampled_addmm(None, device, dtype, requires_grad=True))

        for sample, dense_covector in zip(samples, [True, False]):
            c = sample.input
            a = sample.args[0]
            b = sample.args[1]

            # Compute sparse result
            output = torch.sparse.sampled_addmm(c, a, b, **sample.kwargs)
            covector = torch.randn_like(output).to_dense() if dense_covector else torch.randn_like(output)
            output.backward(covector)

            # Compute dense result and compare with sparse result
            c1, a1, b1 = map(lambda x: x.detach().to_dense().requires_grad_(True), [c, a, b])
            dense_output = sample.kwargs['alpha'] * (a1 @ b1) * torch.ones_like(c).to_dense() + sample.kwargs['beta'] * c1
            self.assertEqual(output, dense_output)
            dense_covector = covector.to_dense()
            dense_output.backward(dense_covector)
            self.assertEqual(c.grad, c1.grad)
            self.assertEqual(a.grad, a1.grad)
            self.assertEqual(b.grad, b1.grad)

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

    @onlyCUDA
    @skipCUDAIf(
        not (TEST_WITH_ROCM or _check_cusparse_sddmm_available()),
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
            torch.sparse.sampled_addmm(a_sparse, a[..., 0, :], a)

        # mat2 must be a matrix
        with self.assertRaisesRegex(RuntimeError, r"Expected mat2 to be a matrix"):
            torch.sparse.sampled_addmm(a_sparse, a, a[..., 0, :])

        a = make_tensor((2, 2), dtype=dtype, device=device)
        b = make_tensor((3, 3), dtype=dtype, device=device)
        b_sparse = b.to_sparse_csr()
        with self.assertRaisesRegex(RuntimeError, r"self.shape\[-2\] must match mat1.shape\[-2\]"):
            torch.sparse.sampled_addmm(b_sparse, a, a)

        b = make_tensor((2, 3), dtype=dtype, device=device)
        b_sparse = b.to_sparse_csr()
        with self.assertRaisesRegex(RuntimeError, r"self.shape\[-1\] must match mat2.shape\[-1\]"):
            torch.sparse.sampled_addmm(b_sparse, a, a)

        a = make_tensor((2, 2), dtype=dtype, device=device)
        a_sparse = a.to_sparse_csr()
        with self.assertRaisesRegex(RuntimeError, r"Expected mat1 to have strided layout"):
            torch.sparse.sampled_addmm(a_sparse, a_sparse, a_sparse)

        with self.assertRaisesRegex(RuntimeError, r"Expected mat2 to have strided layout"):
            torch.sparse.sampled_addmm(a_sparse, a, a_sparse)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_coo_csr_conversion(self, device, dtype):
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            coo_sparse = dense.to_sparse()
            csr_sparse = coo_sparse.to_sparse_csr()

            self.assertEqual(csr_sparse.to_dense(), dense)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_csr_coo_conversion(self, device, dtype):
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            csr_sparse = dense.to_sparse_csr()
            coo_sparse = csr_sparse.to_sparse()

            self.assertEqual(coo_sparse.to_dense(), dense)

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

            self.assertEqual(out, expect)

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
            self.assertEqual(actual, expect)

    @ops(sparse_csr_unary_ufuncs, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble])
    def test_autograd_sparse_csr_unary(self, device, dtype, op):
        if op.name not in UNARY_EWISE_CSR_ALLOW_AUTOGRAD:
            self.skipTest(f"Skipped! Unary op {op.name} not supported with CSR input and autograd")

        samples = list(op.sample_inputs(device, dtype))

        # Fail early to prevent silent success with this test
        ndims_equals_2d = (s.input.ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples.")

        for sample in samples:
            sparse_input = sample.input.to_sparse_csr().requires_grad_(True)

            def fn(input):
                output = op.gradcheck_wrapper(op.get_op(), input, *sample.args, **sample.kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            # Compute sparse result
            output = fn(sparse_input)
            covector = torch.randn_like(output)
            output.backward(covector)
            self.assertTrue(torch.is_tensor(sparse_input.grad))
            self.assertTrue(sparse_input.grad.is_sparse_csr)

            # Compute dense result and compare with sparse result
            dense_input = sparse_input.detach().to_dense().requires_grad_(True)
            dense_output = fn(dense_input)
            dense_covector = covector.to_dense()
            dense_output.backward(dense_covector)
            self.assertEqual(sparse_input.grad, dense_input.grad)

    @skipCUDAIfRocm
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    @dtypes(torch.float64)
    def test_autograd_dense_output_addmm(self, device, dtype):
        from torch.testing._internal.common_methods_invocations import sample_inputs_addmm

        samples = list(sample_inputs_addmm(None, device, dtype, requires_grad=True))

        # Fail early to prevent silent success with this test
        ndims_equals_2d = (s.args[0].ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples to convert to sparse.")

        for sample in samples:
            a = sample.args[0].relu().to_sparse_csr()

            # This path tests the autograd path wrt dense inputs
            for addmm in [torch.addmm, torch.sparse.addmm]:

                def fn(c, b):
                    output = addmm(c, a, b, **sample.kwargs)
                    if sample.output_process_fn_grad is not None:
                        return sample.output_process_fn_grad(output)
                    return output

                self.assertTrue(torch.autograd.gradcheck(fn, [sample.input, sample.args[1]], fast_mode=True))

                # noncontiguous
                c = make_tensor(sample.input.shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True)
                b = make_tensor(sample.args[1].shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(fn, [c, b], fast_mode=True))

                # Now test the autograd path wrt sparse inputs
                for reverse in [True, False]:
                    c, b = sample.input, sample.args[1]
                    if reverse and a.shape != b.shape:
                        continue

                    def fn(a):
                        inputs = (c, b, a) if reverse else (c, a, b)
                        output = addmm(*inputs, **sample.kwargs)
                        if sample.output_process_fn_grad is not None:
                            return sample.output_process_fn_grad(output)
                        return output

                    # gradcheck doesn't work for sparse CSR yet, compare against dense path
                    # Compute sparse result
                    a = a.detach().requires_grad_(True)
                    output = fn(a)
                    covector = torch.randn_like(output)
                    output.backward(covector)
                    self.assertTrue(torch.is_tensor(a.grad))
                    if addmm == torch.sparse.addmm:
                        self.assertTrue(a.grad.is_sparse_csr)
                    else:
                        self.assertTrue(a.grad.layout == torch.strided)

                    # Compute dense result and compare with sparse result
                    dense_a = a.detach().to_dense().requires_grad_(True)
                    dense_output = fn(dense_a)
                    self.assertEqual(output, dense_output)
                    dense_covector = covector.to_dense()
                    dense_output.backward(dense_covector)

                    if addmm == torch.sparse.addmm:
                        self.assertEqual(a.grad, dense_a.grad.sparse_mask(a))
                    else:
                        self.assertEqual(a.grad, dense_a.grad)

    @skipCUDAIfRocm
    @skipCPUIfNoMklSparse
    @dtypes(torch.float64)
    def test_autograd_dense_output_addmv(self, device, dtype):
        from torch.testing._internal.common_methods_invocations import sample_inputs_addmv

        samples = list(sample_inputs_addmv(None, device, dtype, requires_grad=True))

        # Fail early to prevent silent success with this test
        ndims_equals_2d = (s.args[0].ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples to convert to sparse.")

        for sample in samples:
            # TODO: Remove detach once we have autograd support for CSR input
            a = sample.args[0].to_sparse_csr().detach()

            def fn(c, b):
                output = torch.addmv(c, a, b, **sample.kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            self.assertTrue(torch.autograd.gradcheck(fn, [sample.input, sample.args[1]], fast_mode=True))

            # noncontiguous
            c = make_tensor(sample.input.shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True)
            b = make_tensor(sample.args[1].shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(fn, [c, b], fast_mode=True))

    @ops(binary_ops_with_dense_output, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, ])
    def test_autograd_dense_output(self, device, dtype, op):
        if op.name == "mv" and no_mkl_sparse and self.device_type == 'cpu':
            self.skipTest("MKL Sparse is not available")
        if op.name == "mv" and TEST_WITH_ROCM and self.device_type == 'cuda':
            # mv currently work only on CUDA
            self.skipTest("ROCm is not supported")

        samples = list(op.sample_inputs(device, dtype, requires_grad=True))

        # Fail early to prevent silent success with this test
        ndims_equals_2d = (s.input.ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples.")

        # Here we assume that the signature is op(sparse_input, dense_input) -> dense_output
        for sample in samples:
            # TODO: Remove detach once we have autograd support for CSR input
            sparse_input = sample.input.to_sparse_csr().detach()

            def fn(*args):
                output = op.gradcheck_wrapper(op.get_op(), sparse_input, *args, **sample.kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            self.assertTrue(torch.autograd.gradcheck(fn, sample.args, fast_mode=True))

            # noncontiguous
            args = [make_tensor(a.shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True) for a in sample.args]
            self.assertTrue(torch.autograd.gradcheck(fn, args, fast_mode=True))

    @dtypes(*all_types_and_complex())
    def test_direct_coo_csr_conversion(self, device, dtype):
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            coo_sparse = dense.to_sparse_coo()

            self.assertEqual(coo_sparse.to_sparse_csr().to_sparse_coo(), coo_sparse)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sum(self, device, dtype):
        def run_test(shape, nnz, index_type):
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            self.assertEqual(a.sum(), a.values().sum())
            if dtype in floating_types():
                a.requires_grad_(True)
                a.sum().backward()
                self.assertEqual(a.grad, torch.ones(shape, dtype=dtype, device=device))
        for shape, index_dtype in itertools.product(
                [(10, 5), (10, 10)],
                [torch.int32, torch.int64]):
            run_test(shape, 0, index_dtype)
            run_test(shape, max(shape), index_dtype)
            run_test(shape, shape[0] * shape[1], index_dtype)


    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_transpose(self, device, dtype):

        def run_test(shape, nnz, index_type):
            # CSR
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            self.assertEqual(a.layout, torch.sparse_csr)

            # CSC
            a_t = a.transpose(0, 1)
            self.assertEqual(a_t.layout, torch.sparse_csc)

            # CSR
            a_v = a.transpose(0, 0)
            self.assertEqual(a_v.layout, torch.sparse_csr)

            # CSR again
            a_t_t = a_t.transpose(0, 1)
            self.assertEqual(a_t_t.layout, torch.sparse_csr)

            # TODO: Do we want to extend view properties to members as well?
            # These checks are based on is_view_of from test_view_ops.py
            self.assertTrue(a_t._is_view())
            self.assertTrue(a_v._is_view())
            self.assertTrue(a_t_t._is_view())

            self.assertTrue(a_t._base is a)
            self.assertTrue(a_v._base is a)
            self.assertTrue(a_t_t._base is a)

            self.assertFalse(a_t is a)
            self.assertFalse(a_v is a)
            self.assertFalse(a_t_t is a)

            self.assertEqual(a.to_dense().transpose(0, 1), a_t.to_dense())
            self.assertEqual(a.to_dense(), a_v.to_dense())
            self.assertEqual(a.to_dense(), a_t_t.to_dense())

            with self.assertRaisesRegex(RuntimeError, "torch.transpose_: in-place transposition is not supported"):
                a.transpose_(0, 0)

            with self.assertRaisesRegex(RuntimeError, "torch.transpose_: in-place transposition is not supported"):
                a.transpose_(0, 1)


        for shape, index_dtype in itertools.product(
                [(10, 5), (10, 10)],
                [torch.int32, torch.int64]):
            run_test(shape, 0, index_dtype)
            run_test(shape, max(shape), index_dtype)
            run_test(shape, shape[0] * shape[1], index_dtype)

    # TODO: This is a stopgap for a rigorous extension of our autograd tests
    # to test the functionality of detach
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_exercise_detach(self, device, dtype):
        shape = (3, 3)
        nnz = 4
        for index_dtype in [torch.int32, torch.int64]:
            inp = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            detached_inp = inp.detach()
            self.assertEqual(inp, detached_inp)

    def _convert_to_layout(self, a, target_layout, blocksize=(2, 2)):
        """
        Helper function to call the correct layout conversion
        with reasonable defaults for the block size. Clearly there
        is a need for a to.layout overload.
        """
        if target_layout is torch.sparse_csr:
            result = a.to_sparse_csr()
        elif target_layout is torch.sparse_csc:
            result = a.to_sparse_csc()
        elif target_layout is torch.sparse_bsr:
            result = a.to_sparse_bsr(blocksize)
        elif target_layout is torch.sparse_bsc:
            result = a.to_sparse_bsc(blocksize)
        else:
            raise NotImplementedError(repr(a))
        assert result.layout is target_layout
        # to_sparse_xyz methods use unsafe construction of sparse
        # compressed tensors. Here we explicitly validate the results
        # to make sure that the sparse tensors are consistent with the
        # corresponding sparse tensor invariants.
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[result.layout]
        compressed_indices, plain_indices = compressed_indices_mth(result), plain_indices_mth(result)
        torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, result.values(),
                                                      result.shape, result.layout)
        return result

    def _construct_sp_matrix(self, tensor, layout, blocksize=(2, 2)):
        if tensor.layout in [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.strided]:
            tensor = tensor.to_dense()
        else:
            raise NotImplementedError(repr(tensor))
        if layout is torch.sparse_csr:
            return sp.csr_matrix(tensor.cpu().numpy())
        if layout is torch.sparse_csc:
            return sp.csc_matrix(tensor.cpu().numpy())
        if layout is torch.sparse_bsr:
            return sp.bsr_matrix(tensor.cpu().numpy(), blocksize=blocksize).sorted_indices()
        # No native scipy BSC support?
        raise NotImplementedError(repr(tensor))

    @skipMeta
    @all_sparse_compressed_layouts('to_layout')
    @all_sparse_compressed_layouts('from_layout')
    def test_compressed_layout_conversions_coverage(self, device, from_layout, to_layout):
        """
        This test performs a smoke test for covered conversion and verifies
        that an exception is thrown for unsupported conversions.
        """

        def _to_from_layout(layout_a, layout_b):
            a = make_tensor((6, 10), dtype=torch.float, device=device)
            expect_error = (layout_a in [torch.sparse_csc, torch.sparse_bsc]
                            or layout_b in [torch.sparse_csc, torch.sparse_bsc])
            expect_error = expect_error or (layout_a, layout_b) == (torch.sparse_bsr, torch.sparse_bsr)
            expect_error = expect_error or (layout_a, layout_b) == (torch.sparse_bsr, torch.sparse_csr)
            # CSC to CSR conversion is supported
            if layout_a is torch.sparse_csc and layout_b is torch.sparse_csr:
                expect_error = False
            # CSC to CSC conversion is supported
            if layout_a is torch.sparse_csc and layout_b is torch.sparse_csc:
                expect_error = False
            if expect_error:
                with self.assertRaises(RuntimeError):
                    b = self._convert_to_layout(a, layout_a)
                    self._convert_to_layout(b, layout_b)
            else:
                b = self._convert_to_layout(a, layout_a)
                c = self._convert_to_layout(b, layout_b)
                if (layout_a is not torch.sparse_bsr and layout_b is not torch.sparse_bsr):
                    self.assertEqual(a.to_dense(), c.to_dense())

        _to_from_layout(from_layout, to_layout)

    @skipMeta
    @all_sparse_compressed_layouts()
    def test_dense_to_from_sparse_compressed(self, device, layout):
        """
        This test tests conversion from dense to/from CSR and CSC
        by comparing to SciPy's implementation.

        TODO: Eventually this is meant to be merged into test_compressed_layout_conversions_coverage
        """
        if layout is torch.sparse_bsc:
            # TODO: Remove this once support has been enabled
            return

        shapes = [(6, 10), (0, 10), (6, 0), (0, 0)]

        blocksizes = [(2, 2)]
        if layout is torch.sparse_bsr:
            blocksizes += [(3, 5), (6, 10)]

        for shape, blocksize in itertools.product(shapes, blocksizes):
            dense = make_tensor(shape, dtype=torch.float, device=device)
            dense = dense.relu()  # Introduce some sparsity
            sp_matrix = self._construct_sp_matrix(dense, layout, blocksize=blocksize)
            pt_matrix = self._convert_to_layout(dense, layout, blocksize=blocksize)

            compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]

            self.assertEqual(layout, pt_matrix.layout)
            self.assertEqual(sp_matrix.shape, pt_matrix.shape)
            self.assertEqual(torch.tensor(sp_matrix.indptr, dtype=torch.int64), compressed_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.indices, dtype=torch.int64), plain_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values())

            self.assertEqual(dense, pt_matrix.to_dense())

    @skipMeta
    @all_sparse_compressed_layouts()
    @coalescedonoff
    @dtypes(torch.double)
    def test_sparse_to_sparse_compressed(self, device, dtype, coalesced, layout):
        """
        This test tests conversion from COO to CSR and CSC and CSC to CSR and CSC
        by comparing to SciPy's implementation.

        TODO: Eventually this is meant to be merged into test_compressed_layout_conversions_coverage
        """
        if layout is torch.sparse_bsc:
            # TODO: Remove this once support has been enabled
            return
        if layout is torch.sparse_bsr:
            # TODO: Remove this once support has been enabled
            return

        for shape in [(0, 10), (6, 0), (6, 10), (0, 0)]:
            sparse_dim = 2
            nnz = shape[0] * shape[1] // 2
            sparse, _, _ = self.genSparseTensor(shape, sparse_dim, nnz, coalesced, device, dtype)
            sp_matrix = self._construct_sp_matrix(sparse, layout)
            pt_matrix = self._convert_to_layout(sparse, layout)

            compressed_indices_mth = {
                torch.sparse_csr: torch.Tensor.crow_indices,
                torch.sparse_csc: torch.Tensor.ccol_indices,
            }[layout]

            plain_indices_mth = {
                torch.sparse_csr: torch.Tensor.col_indices,
                torch.sparse_csc: torch.Tensor.row_indices,
            }[layout]

            self.assertEqual(layout, pt_matrix.layout)
            self.assertEqual(sp_matrix.shape, pt_matrix.shape)
            self.assertEqual(torch.tensor(sp_matrix.indptr, dtype=torch.int64), compressed_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.indices, dtype=torch.int64), plain_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values())

            sparse_csc = sparse.to_sparse_csc()
            sp_matrix = self._construct_sp_matrix(sparse_csc, layout)
            pt_matrix = self._convert_to_layout(sparse_csc, layout)

            self.assertEqual(layout, pt_matrix.layout)
            self.assertEqual(sp_matrix.shape, pt_matrix.shape)
            self.assertEqual(torch.tensor(sp_matrix.indptr, dtype=torch.int64), compressed_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.indices, dtype=torch.int64), plain_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values())


# e.g., TestSparseCSRCPU and TestSparseCSRCUDA
instantiate_device_type_tests(TestSparseCSR, globals())
instantiate_device_type_tests(TestSparseCompressed, globals())

if __name__ == '__main__':
    run_tests()
