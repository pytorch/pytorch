# Owner(s): ["module: sparse"]

import torch
import random
import itertools
import unittest
import functools
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC
from torch.testing._internal.common_utils import \
    (TEST_WITH_TORCHINDUCTOR, TEST_WITH_ROCM, TEST_SCIPY, TEST_NUMPY, TEST_MKL, IS_WINDOWS, TestCase, run_tests,
     load_tests, coalescedonoff, parametrize, subtest, skipIfTorchDynamo, skipIfRocm, IS_FBCODE, IS_REMOTE_GPU)
from torch.testing._internal.common_device_type import \
    (ops, instantiate_device_type_tests, dtypes, OpDTypes, dtypesIfCUDA, onlyCPU, onlyCUDA, skipCUDAIfNoSparseGeneric,
     precisionOverride, skipMeta, skipCUDAIf, skipCUDAIfRocm, skipCPUIfNoMklSparse, skipCUDAIfRocmVersionLessThan,
     largeTensorTest)
from torch.testing._internal.common_methods_invocations import \
    (op_db, sparse_csr_unary_ufuncs, ReductionOpInfo)
from torch.testing._internal.common_cuda import _get_torch_cuda_version, TEST_CUDA
from torch.testing._internal.common_dtype import (
    floating_types, all_types_and_complex_and, floating_and_complex_types, floating_types_and,
    all_types_and_complex, floating_and_complex_types_and)
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse
from test_sparse import CUSPARSE_SPMM_COMPLEX128_SUPPORTED

if TEST_SCIPY:
    import scipy.sparse as sp

if TEST_NUMPY:
    import numpy as np
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
    return not TEST_WITH_ROCM

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
    'deg2rad',
    'neg',
    'positive',
    'frac',
    'nn.functional.relu',
    'log1p',
    'rad2deg'
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
        elif layout == torch.sparse_csc:
            return mat.to_sparse_csc()
        else:
            assert mat.layout == layout
            return mat

    if mode == "all_sparse":
        res1 = f(*map(convert_layout, (t, m, v)), alpha=alpha, beta=beta)
        test_case.assertEqual(res1.layout, layout)
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


def batched_nonbatched(test_name='batched'):
    return parametrize(test_name, [
        subtest(True, name="Batched"),
        subtest(False, name="NonBatched")
    ])


def hybrid_nonhybrid(test_name='hybrid'):
    return parametrize(test_name, [
        subtest(True, name="Hybrid"),
        subtest(False, name="NonHybrid")
    ])


class TestSparseCompressed(TestCase):
    """Testing sparse compressed (CSR, CSC, BSR, BSC) tensor generic features.
    """

    def genTensor(self, size, nnz, *, layout, device=None, dtype=torch.float, index_dtype=torch.int64):
        if device is None:
            device = self.device_type
        return self.genSparseCompressedTensor(size, nnz, device=device, dtype=dtype, index_dtype=index_dtype, layout=layout)

    @all_sparse_compressed_layouts()
    @onlyCPU
    def test_layout(self, layout):
        self.assertIn(str(layout), {'torch.sparse_csr', 'torch.sparse_csc', 'torch.sparse_bsr', 'torch.sparse_bsc'})
        self.assertEqual(type(layout), torch.layout)

    @parametrize('shape_and_device_inference', [subtest(False, name='_'), subtest(True, name='shape_and_device_inference')])
    @parametrize('use_factory_function', [subtest(False, name='_'), subtest(True, name='factory')])
    @parametrize('input_kind', [subtest('tensor', name='from_tensor'), subtest('list', name='from_list')])
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_compressed_constructor(self, layout, device, dtype,
                                           use_factory_function, shape_and_device_inference, input_kind):
        if input_kind == 'list' and shape_and_device_inference:
            if torch.device(device).type == 'cuda':
                # list inputs to factory/constructor function without
                # specifying device will result a sparse compressed tensor
                # on CPU. So, skip testing against cuda device as unused.
                self.skipTest("nothing to test")
            if dtype not in {torch.float32, torch.complex64, torch.int64, torch.bool}:
                self.skipTest("dtype not supported with list values")

        expected_devices = [torch.device(device)]
        if TEST_CUDA and torch.device(device).type == 'cuda' and torch.cuda.device_count() >= 2 and not shape_and_device_inference:
            expected_devices.append(torch.device('cuda:1'))

        factory_function = {
            torch.sparse_csr: torch.sparse_csr_tensor,
            torch.sparse_csc: torch.sparse_csc_tensor,
            torch.sparse_bsr: torch.sparse_bsr_tensor,
            torch.sparse_bsc: torch.sparse_bsc_tensor,
        }[layout]
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        if input_kind == 'list':
            index_dtypes = [torch.int64]
        else:
            index_dtypes = [torch.int32, torch.int64]
        for index_dtype in index_dtypes:
            for expected_device in expected_devices:
                for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(
                        layout, device=expected_device, dtype=dtype, index_dtype=index_dtype,
                        # skip zero-sized tensors for list inputs:
                        enable_zero_sized=input_kind != 'list',
                        output_tensor=False):
                    size = kwargs['size']
                    if shape_and_device_inference and 0 in size:
                        # skip shape inference for zero-sized tensor
                        # inputs because (i) the shape determined from
                        # an empty list is ambiguous, and (ii) the
                        # size of the plain dimension defined as
                        # max(plain_indices) is undefined if
                        # plain_indices has no values
                        continue
                    compressed_indices_expect = compressed_indices
                    plain_indices_expect = plain_indices
                    values_expect = values

                    if input_kind == 'list':
                        compressed_indices = compressed_indices.tolist()
                        plain_indices = plain_indices.tolist()
                        values = values.tolist()

                    if use_factory_function:
                        if shape_and_device_inference:
                            sparse = factory_function(compressed_indices, plain_indices, values)
                        else:
                            sparse = factory_function(compressed_indices, plain_indices, values, size,
                                                      dtype=dtype, device=expected_device)
                    else:
                        if shape_and_device_inference:
                            sparse = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=layout)
                        else:
                            sparse = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size,
                                                                    dtype=dtype, layout=layout, device=expected_device)
                    self.assertEqual(layout, sparse.layout)
                    self.assertEqual(size, sparse.shape)
                    self.assertEqual(compressed_indices_expect, compressed_indices_mth(sparse))
                    self.assertEqual(plain_indices_expect, plain_indices_mth(sparse))
                    self.assertEqual(values_expect, sparse.values())
                    self.assertEqual(sparse.device, sparse.values().device)
                    self.assertEqual(sparse.device, expected_device)

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
        for sparse in self.generate_simple_inputs(
                layout, device=device, dtype=dtype, index_dtype=torch.int32):
            cloned_sparse = sparse.clone()
            self.assertEqual(sparse, cloned_sparse)

    @all_sparse_compressed_layouts()
    def test_print(self, layout, device):
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        printed = []
        for enable_hybrid in [False, True]:
            # using local patterns for test_print stability
            patterns = [
                # 2 x 3 batch of 3 x 2 tensors, trivial blocksize, non-hybrid/hybrid:
                ([[[[1, 2, 0],
                    [1, 0, 3]],
                   [[1, 2, 3],
                    [1, 0, 0]],
                   [[1, 0, 0],
                    [1, 2, 3]]],
                  [[[0, 2, 0],
                    [1, 2, 3]],
                   [[1, 0, 3],
                    [1, 2, 0]],
                   [[1, 2, 3],
                    [0, 2, 0]]]], [(2, 1)], [(), (4,)] if enable_hybrid else [()]),
                # tensor with non-trivial blocksize, non-hybrid/hybrid:
                ([[0, 1, 0, 2, 0, 2],
                  [0, 1, 0, 0, 2, 0],
                  [3, 3, 3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 5, 0, 6, 6, 6],
                  [5, 0, 5, 6, 6, 6],
                  [0, 0, 0, 0, 8, 8],
                  [7, 7, 7, 0, 8, 8]], [(2, 3)], [(), (4, 2)] if enable_hybrid else [()]),
            ]
            for index_dtype in [torch.int32, torch.int64]:
                for dtype in [torch.float32, torch.float64]:
                    for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(
                            layout, device=device, dtype=dtype, index_dtype=index_dtype, enable_hybrid=enable_hybrid,
                            enable_non_contiguous_indices=False, enable_non_contiguous_values=False,
                            enable_zero_sized=False, output_tensor=False, patterns=patterns):
                        size = tuple(kwargs['size'])
                        block_ndim = 2 if layout in {torch.sparse_bsr, torch.sparse_bsc} else 0
                        base_ndim = 2
                        batch_ndim = compressed_indices.dim() - 1
                        dense_ndim = values.dim() - batch_ndim - block_ndim - 1
                        if enable_hybrid and dense_ndim == 0:
                            # non-hybrid cases are covered by the enable_hybrid==False loop
                            continue
                        batchsize = size[:batch_ndim]
                        basesize = size[batch_ndim:batch_ndim + base_ndim]
                        densesize = size[batch_ndim + base_ndim:]
                        assert len(densesize) == dense_ndim
                        printed.append("########## {}/{}/size={}+{}+{} ##########".format(
                            dtype, index_dtype, batchsize, basesize, densesize))
                        x = torch.sparse_compressed_tensor(compressed_indices,
                                                           plain_indices,
                                                           values, size, dtype=dtype, layout=layout, device=device)
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

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @all_sparse_compressed_layouts()
    @ops(_sparse_compressed_ops)
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_consistency(self, layout, device, dtype, op):
        """Checks that the op on a strided and on a sparse tensors will
        produce the same results.
        """
        if not op.supports_sparse_layout(layout):
            self.skipTest(f"{op.name} does not support input with {layout} layout")

        # FIXME: remove in followup once integer support is landed for segment_reduce
        if (layout == torch.sparse_csr and not dtype.is_floating_point
                and op.name in ('masked.mean', 'masked.amax', 'masked.amin')):
            self.skipTest(f"{op.name} does not support input with {layout} layout and {dtype} dtype")

        require_mask = isinstance(op, ReductionOpInfo) and 'masked.' in op.name

        samples = []
        for sample in op.sample_inputs(device, dtype):
            if sample.input.ndim < 2:
                continue
            dense_dim = sample.input.ndim - 2
            blocksize = (tuple(map(self._smallest_divisor, sample.input.shape[:2]))
                         if layout in {torch.sparse_bsr, torch.sparse_bsc} else None)

            def _to_sparse(x):
                if isinstance(x, torch.Tensor):
                    if blocksize is None:
                        if x.ndim != sample.input.ndim:
                            return x
                    elif x.ndim != sample.input.ndim + 2 or x.shape[-3] % blocksize[0] or x.shape[-2] % blocksize[1]:
                        return x
                    return x.clone().to_sparse(layout=layout, blocksize=blocksize, dense_dim=dense_dim)
                return x

            sparse_sample = sample.transform(_to_sparse)
            # Some strided samples (with inf, nan elements) appear to share
            # storage, so we must clone:
            sample = sample.transform(lambda x: (x.clone() if isinstance(x, torch.Tensor) else x))

            if validate_sample_input_sparse(op, sparse_sample, check_validate=False) is not sparse_sample:
                # that is, the validation returns the sparse sample
                # wrapped within ErrorInput instance
                continue
            samples.append((sample, sparse_sample))

        # Fail early to prevent silent success with this test
        if len(samples) == 0:
            raise ValueError("Expected at least one 2 or higher D tensor in samples.")

        # Re-define atol and rtol for operations that result values
        # are random (and hence, non-comparable) be we still want to
        # check the shape, dtype, etc attributes of the results:
        atol = rtol = None
        if op.name == 'randn_like':
            atol = 1e300
            rtol = 1

        for sample, sparse_sample in samples:
            expected = op(sample.input, *sample.args, **sample.kwargs)
            assert torch.is_tensor(expected)
            output = op(sparse_sample.input, *sparse_sample.args, **sparse_sample.kwargs)
            assert torch.is_tensor(output)
            strided_output = output.to_dense()
            if require_mask and sample.kwargs.get('mask') is not None:
                output_mask = torch.masked._output_mask(op.op, sample.input, *sample.args, **sample.kwargs)
                expected.masked_fill_(~output_mask, 0)
            self.assertEqual(strided_output, expected, atol=atol, rtol=rtol)

    @skipMeta
    @all_sparse_compressed_layouts()
    @all_sparse_compressed_layouts('layout2')
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_empty_like(self, layout, layout2, device, dtype):
        for sparse in self.generate_simple_inputs(layout):
            if layout == layout2:
                result = torch.empty_like(sparse, layout=layout2)
                compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[result.layout]
                torch._validate_sparse_compressed_tensor_args(compressed_indices_mth(result),
                                                              plain_indices_mth(result),
                                                              result.values(),
                                                              result.shape,
                                                              result.layout)
                self.assertEqual(sparse.shape, result.shape)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "empty_like with different sparse layout is not supported",
                    lambda: torch.empty_like(sparse, layout=layout2)
                )

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_validate(self, layout, device, dtype):
        def make_zero_batched(t):
            return torch.empty(*((0,) + t.shape), dtype=t.dtype, device=t.device)

        for index_dtype in [torch.int32, torch.int64]:
            for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(
                    layout, device=device, dtype=dtype, index_dtype=index_dtype, output_tensor=False):
                size = kwargs['size']
                torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, values, size, layout)

                # check empty batch
                torch._validate_sparse_compressed_tensor_args(
                    *(make_zero_batched(t) for t in (compressed_indices, plain_indices, values)),
                    (0,) + size,
                    layout
                )

    def _generate_invalid_input(self, layout, device):
        from functools import partial

        def shape(shape, basedim=0):
            blocksize = (1, 1)
            if layout is torch.sparse_csc:
                shape = shape[:basedim] + (shape[basedim + 1], shape[basedim]) + shape[basedim + 2:]
            elif layout is torch.sparse_bsc:
                shape = shape[:basedim] + (shape[basedim + 1] * blocksize[1], shape[basedim] * blocksize[0]) + shape[basedim + 2:]
            elif layout is torch.sparse_bsr:
                shape = shape[:basedim] + (shape[basedim] * blocksize[0], shape[basedim + 1] * blocksize[1]) + shape[basedim + 2:]
            return shape

        def values(lst, device=device):
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                lst = [[[item]] for item in lst]
            return torch.tensor(lst, device=device)

        tensor = partial(torch.tensor, device=device)
        values = partial(values, device=device)

        yield ('incontiguous compressed_indices',
               tensor([0, -1, 2, -1, 4, -1])[::2],
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'expected compressed_indices to be a contiguous tensor per batch')

        yield ('incontiguous plain_indices',
               tensor([0, 2, 4]),
               tensor([0, -1, 1, -1, 0, -1, 2, -1])[::2],
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'expected plain_indices to be a contiguous tensor per batch')

        yield ('0-D compressed_indices',
               tensor(0),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'compressed_indices must have dimensionality >= 1 but got 0')

        yield ('compressed/plain_indices mismatch of dimensionalities',
               tensor([[0, 2, 4]]),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'compressed_indices and plain_indices dimensionalities must be equal but got 2 and 1, respectively')

        if layout in {torch.sparse_csr, torch.sparse_csc}:
            yield ('indices and values mismatch of dimensionalities',
                   tensor([[0, 2, 4]]),
                   tensor([[0, 1, 0, 2]]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'values must have dimensionality > sum of batch and block dimensionalities \(=1 \+ 0\) but got 1')
        else:
            yield ('indices and values mismatch of dimensionalities',
                   tensor([[0, 2, 4]]),
                   tensor([[0, 1, 0, 2]]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'values must have dimensionality > sum of batch and block dimensionalities \(=1 \+ 2\) but got 3')

        yield ('invalid size',
               tensor([0, 2, 4]),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               (2,),
               r'tensor dimensionality must be sum of batch, base, and dense dimensionalities \(=0 \+ 2 \+ 0\) but got 1')

        yield ('invalid batchsize',
               tensor([[0, 2, 4]]),
               tensor([[0, 1, 0, 2]]),
               values([[1, 2, 3, 4]]),
               shape((2, 2, 3), 1),
               r'all batch dimensions of compressed_indices \(=\[1\]\), plain_indices \(=\[1\]\), '
               r'and values \(=\[1\]\) must be equal to tensor batch dimensions \(=\[2\]\)')

        if layout is torch.sparse_bsr:
            yield ('invalid blocksize',
                   tensor([0, 2, 4]),
                   tensor([0, 1, 0, 2]),
                   tensor([[[1, 11]], [[2, 22]], [[3, 33]], [[4, 33]]]),
                   shape((2, 3)),
                   r'tensor shape\[1\] \(=3\) must be divisible with blocksize\[1\] \(=2\) as defined by values shape')

        if layout is torch.sparse_bsc:
            yield ('invalid blocksize',
                   tensor([0, 2, 4]),
                   tensor([0, 1, 0, 2]),
                   tensor([[[1, 11]], [[2, 22]], [[3, 33]], [[4, 33]]]),
                   shape((3, 2)),
                   r'tensor shape\[1\] \(=3\) must be divisible with blocksize\[1\] \(=2\) as defined by values shape')

        yield ('invalid compressed_indices shape',
               tensor([0, 2, 3, 4]),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'compressed_indices.shape\[-1\] must be equal to the number of compressed_indices_names \+ 1 \(=3\), but got 4')

        yield ('invalid compressed_indices shape',
               tensor([0, 2, 4]),
               tensor([0, 1, 0, 1, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'plain_indices.shape\[-1\] must be equal to nnz \(=4\) as defined by values.shape\[0\], but got 5')

        yield ('compressed/plain_indices mismatch of dtype',
               tensor([0, 2, 4], dtype=torch.int32),
               tensor([0, 1, 0, 2], dtype=torch.int64),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'compressed_indices and plain_indices must have the same dtype, bot got Int and Long, respectively')

        yield ('invalid compressed/plain_indices dtype',
               tensor([0, 2, 4], dtype=torch.int16),
               tensor([0, 1, 0, 2], dtype=torch.int16),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'compressed_indices and plain_indices dtype must be Int or Long, but got Short')

        # CUDA kernel asserts are not recoverable, so we skip these for now
        if torch.device(device).type == 'cpu':
            yield ('invalid compressed_indices[0]',
                   tensor([1, 2, 4]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`compressed_indices\[..., 0\] == 0` is not satisfied.')

            yield ('invalid compressed_indices[-1]',
                   tensor([0, 2, 5]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`compressed_indices\[..., -1\] == nnz` is not satisfied.')

            yield ('invalid compressed_indices.diff(dim=-1)',
                   tensor([0, 0, 4]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'0 <= compressed_indices\[..., 1:\] - compressed_indices\[..., :\-1\] <= plain_dim` is not satisfied.')

            yield ('invalid compressed_indices.diff(dim=-1)',
                   tensor([0, 5, 4]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'0 <= compressed_indices\[..., 1:\] - compressed_indices\[..., :\-1\] <= plain_dim` is not satisfied.')

            yield ('invalid min(plain_indices)',
                   tensor([0, 2, 4]),
                   tensor([0, -1, 0, 3]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`0 <= plain_indices < plain_dim` is not satisfied.')

            yield ('invalid max(plain_indices)',
                   tensor([0, 2, 4]),
                   tensor([0, 1, 0, 3]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`0 <= plain_indices < plain_dim` is not satisfied.')

            yield ('non-coalesced',
                   tensor([0, 2, 4]),
                   tensor([1, 0, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`plain_indices\[..., compressed_indices\[..., i - 1\]:compressed_indices\[..., i\]\] '
                   'for all i = 1, ..., compressed_dim '
                   'are sorted and distinct along the last dimension values` is not satisfied.')

        if TEST_CUDA and torch.device(device).type == 'cpu':
            yield ('indices and values mismatch of device',
                   torch.tensor([0, 2, 4]),
                   torch.tensor([0, 1, 0, 1]),
                   values([1, 2, 3, 4], device='cuda'),
                   shape((2, 3)),
                   r'device of compressed_indices \(=cpu\) must match device of values \(=cuda:0\)')
            yield ('compressed_indices and values mismatch of device',
                   torch.tensor([0, 2, 4], device='cuda'),
                   torch.tensor([0, 1, 0, 1]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
            yield ('compressed/plain_indices mismatch of device',
                   torch.tensor([0, 2, 4], device='cuda'),
                   torch.tensor([0, 1, 0, 1]),
                   values([1, 2, 3, 4], device='cuda'),
                   shape((2, 3)),
                   r'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')

        if TEST_CUDA and torch.device(device).type == 'cuda' and torch.cuda.device_count() >= 2:
            yield ('indices and values mismatch of device index',
                   torch.tensor([0, 2, 4], device='cuda:0'),
                   torch.tensor([0, 1, 0, 1], device='cuda:0'),
                   values([1, 2, 3, 4], device='cuda:1'),
                   shape((2, 3)),
                   r'device of compressed_indices \(=cuda:0\) must match device of values \(=cuda:1\)')
            yield ('compressed_indices and values mismatch of device index',
                   torch.tensor([0, 2, 4], device='cuda:0'),
                   torch.tensor([0, 1, 0, 1], device='cuda:1'),
                   values([1, 2, 3, 4], device='cuda:0'),
                   shape((2, 3)),
                   r'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!')

    @skipMeta
    @all_sparse_compressed_layouts()
    @parametrize('target', [subtest('validate_sparse_compressed_tensor_args'),
                            subtest('sparse_compressed_tensor'),
                            subtest('sparse_compressed_tensor_no_size')])
    def test_invalid_input(self, layout, device, target):
        for label, compressed_indices, plain_indices, values, size, errmsg in self._generate_invalid_input(layout, device):
            if layout is torch.sparse_bsr:
                errmsg = errmsg.replace('compressed_indices_name', 'row block').replace('plain_indices_name', 'column block')
            elif layout is torch.sparse_bsc:
                errmsg = errmsg.replace('compressed_indices_name', 'column block').replace('plain_indices_name', 'row block')
            elif layout is torch.sparse_csr:
                errmsg = errmsg.replace('compressed_indices_name', 'row').replace('plain_indices_name', 'column')
            elif layout is torch.sparse_csc:
                errmsg = errmsg.replace('compressed_indices_name', 'column').replace('plain_indices_name', 'row')
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                errmsg = errmsg.replace('compressed_indices', 'crow_indices') \
                               .replace('plain_indices', 'col_indices') \
                               .replace('plain_dim', 'ncols') \
                               .replace('compressed_dim', 'nrows')
            else:
                errmsg = errmsg.replace('compressed_indices', 'ccol_indices') \
                               .replace('plain_indices', 'row_indices') \
                               .replace('plain_dim', 'nrows') \
                               .replace('compressed_dim', 'ncols')

            if target == 'sparse_compressed_tensor_no_size' and label in {
                    'invalid size', 'invalid batchsize', 'invalid compressed_indices shape', 'invalid max(plain_indices)',
                    'invalid blocksize'}:
                # Skip invalid size input as a valid size is estimated for other inputs
                continue

            with self.assertRaisesRegex(RuntimeError, errmsg):
                if target == 'validate_sparse_compressed_tensor_args':
                    torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, values, size, layout)
                elif target == 'sparse_compressed_tensor':
                    torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size, layout=layout)
                elif target == 'sparse_compressed_tensor_no_size':
                    torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=layout)
                else:
                    raise NotImplementedError(target)

    @skipMeta
    @onlyCPU
    @largeTensorTest("30GB", "cpu")
    def test_invalid_input_csr_large(self):
        rows = 2 ** 31
        with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in row dimension'):
            torch.sparse_csr_tensor(torch.arange(rows + 1, dtype=torch.int32) // rows,
                                    torch.tensor([0], dtype=torch.int32),
                                    torch.tensor([1]), (rows, 1))
        torch.sparse_csr_tensor(torch.arange(rows + 1, dtype=torch.int64) // rows,
                                torch.tensor([0], dtype=torch.int64),
                                torch.tensor([1]), (rows, 1))

        cols = 2 ** 31
        with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in column dimension'):
            torch.sparse_csr_tensor(torch.arange(2, dtype=torch.int32),
                                    torch.tensor([0], dtype=torch.int32),
                                    torch.tensor([1]), (1, cols))
        torch.sparse_csr_tensor(torch.arange(2, dtype=torch.int64),
                                torch.tensor([0], dtype=torch.int64),
                                torch.tensor([1]), (1, cols))

        nnz = 2 ** 31
        with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in nnz'):
            torch.sparse_csr_tensor(torch.tensor([0, nnz // 2, nnz], dtype=torch.int32),
                                    torch.arange(nnz // 2, dtype=torch.int32).repeat(2),
                                    torch.ones(nnz, dtype=torch.int8), (2, nnz // 2))
        torch.sparse_csr_tensor(torch.tensor([0, nnz // 2, nnz], dtype=torch.int64),
                                torch.arange(nnz // 2, dtype=torch.int64).repeat(2),
                                torch.ones(nnz, dtype=torch.int8), (2, nnz // 2))

    @skipMeta
    @onlyCPU
    @all_sparse_compressed_layouts()
    def test_dim(self, layout):
        for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(layout, output_tensor=False):
            size = kwargs['size']
            batch_dim = compressed_indices.dim() - 1
            sparse_dim = 2
            block_dim = 2 if layout in {torch.sparse_bsr, torch.sparse_bsc} else 0
            dense_dim = values.dim() - batch_dim - block_dim - 1
            sparse = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size, layout=layout)
            self.assertEqual(sparse.sparse_dim(), sparse_dim)
            self.assertEqual(sparse.dense_dim(), dense_dim)


    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_to_dtype(self, layout, device, dtype):
        # to_dense does not support hybrid inputs
        for sparse in self.generate_simple_inputs(layout, dtype=dtype, device=device, enable_hybrid=False):
            for to_dtype in all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16):
                sparse_to_dtype = sparse.to(to_dtype)
                dense_to_dtype = sparse.to_dense().to(to_dtype)
                self.assertEqual(sparse_to_dtype.to_dense(), dense_to_dtype)

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(torch.double)
    def test_pickle(self, layout, dtype, device):
        import pickle

        for sparse in self.generate_simple_inputs(layout, device=device, dtype=dtype):
            serialized = pickle.dumps(sparse)
            sparse_loaded = pickle.loads(serialized)

            self.assertEqual(sparse, sparse_loaded)

    @all_sparse_compressed_layouts()
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_select_copy(self, device, dtype, index_dtype, layout):

        def is_view_of(base, other):
            # a shameless copy of TestViewOps.is_view_of
            if (
                not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device
            ):
                return False
            if base.device.type in ('cpu', 'cuda'):
                if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                    return False
            return True

        kwargs = dict(device=device, dtype=dtype, index_dtype=index_dtype)
        for sparse, dense in zip(self.generate_simple_inputs(layout, **kwargs),
                                 self.generate_simple_inputs(torch.strided, **kwargs)):
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                n_batchdim = sparse.crow_indices().ndim - 1
            elif layout in {torch.sparse_csc, torch.sparse_bsc}:
                n_batchdim = sparse.ccol_indices().ndim - 1
            else:
                assert 0  # unreachable
            self.assertEqual(sparse, dense)
            for dim in range(sparse.ndim):
                if sparse.shape[dim] == 0:
                    with self.assertRaisesRegex(IndexError, "index 0 out of range for tensor of size"):
                        torch.select_copy(sparse, dim, 0)
                    with self.assertRaisesRegex(IndexError, "index 0 out of range for tensor of size"):
                        torch.select_copy(dense, dim, 0)
                elif n_batchdim and dim >= n_batchdim and dim < n_batchdim + 2:
                    with self.assertRaisesRegex(
                            RuntimeError,
                            "selecting sparse dimensions is not supported for batched sparse compressed tensors"):
                        torch.select_copy(sparse, dim, 0)
                else:
                    for index in {0, sparse.shape[dim] // 2, sparse.shape[dim] - 1}:
                        dense_select = torch.select_copy(dense, dim, index)
                        sparse_select = torch.select_copy(sparse, dim, index)
                        self.assertEqual(sparse_select, dense_select)
                        self.assertFalse(is_view_of(sparse_select.values(), sparse.values()))


def _npref_block_addmm_addmv(c, a, b, alpha, beta):
    return alpha * (a @ b) + beta * c


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

        with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have is_contiguous"):
            a.is_contiguous()

    @onlyCPU
    @largeTensorTest("20GB", "cpu")
    def test_csr_nnz(self):
        # Tests the limits of the number of specified elements in CSR tensors, see gh-102520.
        for nnz in [0, 2**31]:
            rows, cols = 1, max(nnz, 1)
            crow_indices = torch.tensor([0, nnz], dtype=torch.int64)
            col_indices = torch.arange(nnz, dtype=torch.int64)
            values = torch.ones(nnz, dtype=torch.int8)
            a = torch.sparse_csr_tensor(crow_indices, col_indices, values, (rows, cols))
            self.assertEqual(a._nnz(), nnz)

    def test_csr_double_to_sparse_csr(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)
        a.to_sparse_csr().to_sparse_csr()

    @all_sparse_compressed_layouts()
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_select(self, device, dtype, index_dtype, layout):
        compressed_indices_mth = {
            torch.sparse_csr: torch.Tensor.crow_indices,
            torch.sparse_bsr: torch.Tensor.crow_indices,
            torch.sparse_csc: torch.Tensor.ccol_indices,
            torch.sparse_bsc: torch.Tensor.ccol_indices,
        }[layout]

        plain_indices_mth = {
            torch.sparse_csr: torch.Tensor.col_indices,
            torch.sparse_bsr: torch.Tensor.col_indices,
            torch.sparse_csc: torch.Tensor.row_indices,
            torch.sparse_bsc: torch.Tensor.row_indices,
        }[layout]
        create_tensor_mth = {
            torch.sparse_csr: torch.sparse_csr_tensor,
            torch.sparse_bsr: torch.sparse_bsr_tensor,
            torch.sparse_csc: torch.sparse_csc_tensor,
            torch.sparse_bsc: torch.sparse_bsc_tensor,
        }[layout]

        shape = (2, 3, 6, 10)
        nnz = 6
        blocksize = (2, 2) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        sparse = self.genSparseCompressedTensor(
            shape, nnz, device=device, layout=layout, dtype=dtype, index_dtype=index_dtype, blocksize=blocksize)
        comp_indices = compressed_indices_mth(sparse)
        plain_indices = plain_indices_mth(sparse)
        values = sparse.values()

        # select from batch dimensions
        sparse_selected12 = sparse.select(1, 2)
        expected_sparse_selected12 = create_tensor_mth(comp_indices.select(1, 2).contiguous(),
                                                       plain_indices.select(1, 2).contiguous(),
                                                       values.select(1, 2).contiguous(),
                                                       size=(2, 6, 10),
                                                       dtype=dtype,
                                                       device=device)
        self.assertEqual(expected_sparse_selected12, sparse_selected12)

        # selecting rows/col with batch dims not allowed
        sparse_non_batched = sparse[0, 0]
        # select from sparse dimensions
        for select_args in [(0, 0), (1, 1)]:
            sparse_selected = sparse_non_batched.select(*select_args)
            dense_selected = sparse_non_batched.to_dense().select(*select_args)
            self.assertEqual(dense_selected, sparse_selected)

        self.assertEqual(sparse[0, 0, 0, 0], sparse.to_dense()[0, 0, 0, 0])
        # assigning to sparse through indexing is disabled
        with self.assertRaisesRegex(TypeError, "Cannot assign to a sparse tensor"):
            sparse[0, 0, 0, 0] = 99.0

        # select from sparse dimensions without removing batch dims
        msg = "selecting sparse dimensions is not supported for batched sparse compressed tensors."
        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-2, 0)

        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-1, 0)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_resize(self, device, dtype):

        def numel(tensor):
            r = 1
            for s in tensor.shape:
                r *= s
            return r

        batch_shapes = [(), (2,), (2, 3)]
        for index_dtype, b in zip([torch.int32, torch.int64], batch_shapes):
            shape = (*b, 2, 3)
            nnz = 6
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            self.assertEqual(a.numel(), numel(a))

            new_shape = (*b, 4, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to larger shape doesn't add specified elements
            self.assertEqual(a._nnz(), nnz)
            self.assertEqual(a.numel(), numel(a))

            new_shape = (*b, 1, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to smaller shape trims specified elements
            self.assertEqual(a._nnz(), 5)
            self.assertEqual(a.numel(), numel(a))

            # trim batched dimensions
            a.resize_(new_shape[-2], new_shape[-1])
            self.assertEqual(a.shape, (new_shape[-2], new_shape[-1]))
            self.assertEqual(a._nnz(), 5)
            self.assertEqual(a.numel(), numel(a))

    @skipMeta
    @dtypes(torch.float, torch.bool)
    @all_sparse_compressed_layouts()
    def test_resize_as_sparse_compressed(self, device, dtype, layout):

        def _check_resize_b_as_a(b, a):
            br = b.clone()
            br.resize_as_sparse_(a)

            # shape is inherited from a
            self.assertEqual(a.shape, br.shape)
            # other metadata is not affected
            self.assertEqual(b.layout, br.layout)
            self.assertEqual(b.device, br.device)
            self.assertEqual(b.dtype, br.dtype)

            def _get_compressed_plain_inds(t):
                compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[t.layout]
                return compressed_indices_mth(t), plain_indices_mth(t)

            br_compressed_indices, br_plain_indices = _get_compressed_plain_inds(br)
            br_values = br.values()

            b_compressed_indices, b_plain_indices = _get_compressed_plain_inds(b)
            a_compressed_indices, a_plain_indices = _get_compressed_plain_inds(a)
            self.assertEqual(a_plain_indices.shape, br_plain_indices.shape)
            self.assertEqual(a_compressed_indices.shape, br_compressed_indices.shape)
            # We don't check the content of br_plain_indices and br_compressed_indices
            # because it is not well-defined (the content depends on the original
            # shape of `b` that `resize_as` ought to discard) nor needed (the
            # subsequent operation likely updates the indices and values of `b` anyway).
            # the device/dtype of indices should always be unaffected
            self.assertEqual(b_plain_indices.dtype, br_plain_indices.dtype)
            self.assertEqual(b_plain_indices.device, br_plain_indices.device)
            self.assertEqual(b_compressed_indices.dtype, br_compressed_indices.dtype)
            self.assertEqual(b_compressed_indices.device, br_compressed_indices.device)
            # values are generated empty, shape is updated
            self.assertEqual(a.values().shape, br_values.shape)
            # the device/dtype of indices should always be unaffected
            b_values = b.values()
            self.assertEqual(b_values.dtype, br_values.dtype)
            self.assertEqual(b_values.device, br_values.device)
            # nnz will be picked up from a via new shape of values
            self.assertEqual(a._nnz(), br._nnz())

            # post resize the invariants of the layout are respected
            torch._validate_sparse_compressed_tensor_args(br_compressed_indices, br_plain_indices, br_values, br.shape,
                                                          br.layout)

        block_sparse = layout in (torch.sparse_bsr, torch.sparse_bsc)
        shape = (2, 1, 6, 4)
        nnz = 4
        blocksize = (2, 1) if block_sparse else ()
        for index_dtype in [torch.int32, torch.int64]:
            a = self.genSparseCompressedTensor(shape,
                                               layout=layout,
                                               device=device,
                                               index_dtype=index_dtype,
                                               dtype=dtype,
                                               nnz=nnz,
                                               blocksize=blocksize)

            # same size, resize should not trigger
            b = self.genSparseCompressedTensor(shape,
                                               layout=layout,
                                               device=device,
                                               index_dtype=index_dtype,
                                               dtype=dtype,
                                               nnz=nnz,
                                               blocksize=blocksize)

            # This test will not always trigger a resize, if the layouts are the same nothing should happen to b.
            # The invariants of the function as checked should still hold
            _check_resize_b_as_a(b, a)

            # same ndim, but bigger, more nnz, different dtype, different blocksize if blocked
            b = self.genSparseCompressedTensor(tuple(s * 2 for s in shape),
                                               layout=layout,
                                               device=device,
                                               dtype=torch.chalf,
                                               index_dtype=torch.int64 if index_dtype == torch.int32 else torch.int32,
                                               nnz=nnz * 2,
                                               blocksize=tuple(2 * bi for bi in blocksize))
            _check_resize_b_as_a(b, a)

            # different device, only check on cuda pass as we know we are testing in an environment
            # that has multiple devices

            # TODO: .cpu() does not seem to work correctly for sparse. Causes a call to `copy_` which
            # complains about incompatible nnz between src and self?
            if torch.device(device).type == 'cuda' and (layout not in (torch.sparse_bsc, torch.sparse_bsr)):
                a_cpu = self.genSparseCompressedTensor(shape,
                                                       layout=layout,
                                                       device='cpu',
                                                       index_dtype=index_dtype,
                                                       dtype=dtype,
                                                       nnz=nnz,
                                                       blocksize=blocksize)
                _check_resize_b_as_a(b, a)

            # error on a strided
            a_strided = a.to_dense()
            with self.assertRaisesRegex(
                    RuntimeError, r'resize_as_sparse_compressed_: src  expected sparse compressed tensor layout'):
                b.resize_as_sparse_(a_strided)

            # error on b strided
            b_strided = b.to_dense()
            with self.assertRaisesRegex(
                    RuntimeError, r'resize_as_sparse_compressed_: self  expected sparse compressed tensor layout'):
                b_strided.resize_as_sparse_(a)

            # error if layout does not match, transpose induces layout flip
            with self.assertRaisesRegex(RuntimeError,
                                        r"resize_as_sparse_compressed_tensor_: self and src must have the same layout"):
                b.transpose(-2, -1).resize_as_sparse_(a)
            with self.assertRaisesRegex(RuntimeError,
                                        r"resize_as_sparse_compressed_tensor_: self and src must have the same layout"):
                b.resize_as_sparse_(a.transpose(-2, -1))

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
            block_st.sort_indices()
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

            with self.assertRaisesRegex(RuntimeError,
                                        r"tensor sparse size \(.*,.*\) must be divisible by given blocksize \(.*,.*\)"):
                block_t = t.to_sparse_bsr((5, 5))

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
    @skipCUDAIfNoSparseGeneric
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater else [],
                  *[torch.bfloat16] if SM80OrLater else []))
    def test_csr_matvec(self, device, dtype):

        if TEST_WITH_ROCM and (dtype == torch.half or dtype == torch.bfloat16):
            self.skipTest("ROCm doesn't work with half dtypes correctly.")

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
    # hmm, the test passes ok on CUDA when Rocm is not available:
    @skipCUDAIfRocmVersionLessThan((5, 2))
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_baddbmm(self, device, dtype):

        # TODO: disable the invariant checks within torch.baddbmm that
        # constructs unconventional csr tensors leading to
        # RuntimeError: tensor dimensionality must be sum of batch,
        #     base, and dense dimensionalities (=0 + 2 + 0) but got 3
        # when invariant checking is enabled. When done, undecorate run_test.
        @torch.sparse.check_sparse_tensor_invariants(enable=False)
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
                a_batched = torch.sparse_csr_tensor(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k), check_invariants=False)

                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                c = make_tensor((batch_size, m, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    run_test(c, a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    @onlyCUDA
    @unittest.skipIf(TEST_WITH_ROCM, "Only CUDA 11+ is supported")
    @skipCUDAIfNoSparseGeneric
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

                # a_batched is a regular CSR tensor but with a batch
                # dimension in the shape. It is unorthodox in PyTorch
                # to represent a batch sparse tensor in this way,
                # hence checking the tensor invariants is locally
                # turned off.
                a_batched = torch.sparse_csr_tensor(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k), check_invariants=False)

                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    run_test(a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    def run_test_block_addmm_addmv(self,
                                   addmv_addmm,
                                   c,
                                   a,
                                   b,
                                   op_b=False,
                                   op_out=False,
                                   *,
                                   dtype=None,
                                   device=None,
                                   ref=_npref_block_addmm_addmv):
        alpha = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        beta = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        b = b.mH if (op_b and a.shape == b.shape) else b

        actual = addmv_addmm(c, a, b, alpha=alpha, beta=beta)

        out = torch.empty_like(c.mH if op_out and a.shape == b.shape else c)
        addmv_addmm(c, a, b, alpha=alpha, beta=beta, out=out)
        expected = ref(c, a, b, alpha, beta)

        self.assertEqual(actual, out)
        self.assertEqual(actual, expected, lambda msg: f"{msg}\na={a}\nc={c}\nb={b}\nalpha={alpha} beta={beta}")

    # TODO: block_size 1 is broken
    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater else [],
                  *[torch.bfloat16] if SM80OrLater else []))
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-5, torch.complex128: 1e-5,
                        torch.float16: 1e-3, torch.bfloat16: 1e-3})
    def test_block_addmm(self, device, dtype, index_dtype, block_size, noncontiguous):

        def make_transposed_addmm_op(f):

            def tt(t):
                if isinstance(t, torch.Tensor):
                    return t.transpose(-2, -1)
                else:
                    # assume numpy/scipy spmatrix
                    return t.transpose()

            @functools.wraps(f)
            def wrapper(c, a, b, alpha=None, beta=None, out=None):
                if out is not None:
                    # the ref takes no out kwarg
                    assert isinstance(out, torch.Tensor)
                    # transpose inplace to propagate out to checking context
                    out.transpose_(-2, -1)
                    return f(tt(c), tt(b), tt(a), alpha=alpha, beta=beta, out=out)
                else:
                    return f(tt(c), tt(b), tt(a), alpha=alpha, beta=beta)

            return wrapper

        def ref_sp_numpy(c, a, b, alpha=None, beta=None, out=None):

            def prep_input(t):

                def to_sp_block_compressed(t):

                    if t.layout is torch.sparse_bsc:
                        tt = t.transpose(-1, -2)
                    else:
                        tt = t

                    t_sp_bsr = sp.bsr_matrix(
                        (
                            tt.values().cpu().numpy(),
                            tt.col_indices().cpu().numpy(),
                            tt.crow_indices().cpu().numpy(),
                        ),
                        shape=tt.shape,
                    )

                    if t.layout is torch.sparse_bsc:
                        return t_sp_bsr.transpose()
                    else:
                        return t_sp_bsr

                if t.layout is not torch.strided:
                    return to_sp_block_compressed(t)
                else:
                    return t.cpu().resolve_conj().numpy()

            res = _npref_block_addmm_addmv(
                *(prep_input(t) for t in (c, a, b)),
                alpha,
                beta
            )

            if out is not None:
                out.copy_(res)
                return out
            else:
                return res

        def ref_half_bfloat16(c, a, b, alpha=None, beta=None, out=None):
            res = alpha * (a.to_dense().to(torch.float32) @ b.to_dense().to(torch.float32)).to(a.dtype) + beta * c
            if out is not None:
                out.copy_(res)
                return out
            else:
                return res

        if dtype in (torch.half, torch.bfloat16):
            ref = ref_half_bfloat16
        else:
            ref = ref_sp_numpy

        for (m, n, k) in itertools.product([2, 5], repeat=3):
            nnz = random.randint(0, m * k)
            a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
            a_data = a_data.mT if noncontiguous else a_data
            a = torch.sparse_bsr_tensor(a.crow_indices(), a.col_indices(),
                                        a_data, (m * block_size, k * block_size), check_invariants=False)
            b = make_tensor((k * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
            c = make_tensor((m * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
            for op_b, op_out in itertools.product([True, False], repeat=2):
                self.run_test_block_addmm_addmv(torch.addmm, c, a, b, op_b, op_out, dtype=dtype, device=device, ref=ref)
                self.run_test_block_addmm_addmv(make_transposed_addmm_op(torch.addmm),
                                                c,
                                                a,
                                                b,
                                                op_b,
                                                op_out,
                                                dtype=dtype,
                                                device=device,
                                                ref=make_transposed_addmm_op(ref))

    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_addmv(self, device, dtype, index_dtype, block_size, noncontiguous):
        # TODO: Explicitly disable block size 1 support
        # if (TEST_WITH_ROCM or not TEST_CUSPARSE_GENERIC) and block_size == 1:
        #     return
        def ref_block_addmv(c, a, b, alpha, beta):
            return _npref_block_addmm_addmv(c, a.to_dense(), b, alpha, beta)

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
                a = torch.sparse_bsr_tensor(a.crow_indices(), a.col_indices(),
                                            a_data, (m * block_size, k * block_size), check_invariants=False)
            b = make_tensor((k * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            c = make_tensor((m * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            self.run_test_block_addmm_addmv(torch.addmv, c, a, b, dtype=dtype, device=device, ref=ref_block_addmv)

    @parametrize("matrix_shape", [(3, 3), (5, 7), (11, 9)], name_fn=lambda x: "shape_{}x{}".format(*x))
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @onlyCPU
    def test_addmv(self, device, dtype, matrix_shape):
        mat = torch.randn(matrix_shape, dtype=dtype, device=device)
        mat[mat.real < 0] = 0
        sparse_mat = mat.to_sparse_csr()
        mvec = torch.randn((mat.size(1),), dtype=dtype, device=device)
        avec = torch.randn((mat.size(0),), dtype=torch.float64, device=device)
        ref_output = torch.addmv(avec, mat, mvec)
        output = torch.addmv(avec, sparse_mat, mvec)
        self.assertEqual(ref_output, output)

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
                a = torch.sparse_bsr_tensor(a.crow_indices(), a.col_indices(),
                                            a_data, (m * block_size, m * block_size), check_invariants=False)
            b = make_tensor((m * block_size, k), dtype=dtype, device=device, noncontiguous=noncontiguous)

            for (upper, unitriangular, transpose, op_out) in itertools.product([True, False], repeat=4):
                run_test(a, b, upper, unitriangular, transpose, op_out)

    @skipCPUIfNoMklSparse
    @unittest.skipIf(TEST_WITH_ROCM, "Only CUDA 11+ is supported")
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

                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = self.genSparseCSCTensor((di, dk), nnz0, device=device, dtype=dtype, index_dtype=index_dtype)
                y = torch.randn(dk, dj, dtype=dtype, device=device)
                _test(t, x, y)

                if nnz1 is None:
                    nnz1 = random.randint(dk * dj // 2, dk * dj)
                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = torch.randn(di, dk, dtype=dtype, device=device)
                y = self.genSparseCSRTensor((dk, dj), nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                _test(t, x, y)

                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = torch.randn(di, dk, dtype=dtype, device=device)
                y = self.genSparseCSCTensor((dk, dj), nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                _test(t, x, y)

                x_shape, y_shape = x.shape, y.shape

                gen_csr_csc = [self.genSparseCSRTensor, self.genSparseCSCTensor]

                # Test mm({CSR, CSC}, {CSR, CSC})
                for gen_x, gen_y in itertools.product(gen_csr_csc, gen_csr_csc):
                    x = gen_x(x_shape, nnz0, device=device, dtype=dtype, index_dtype=index_dtype)
                    y = gen_y(y_shape, nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                    _test_mm(x, y)

        def test_empty_inputs(lhs_layout, rhs_layout):
            xd = torch.rand(10, 0, device=device, dtype=dtype)
            yd = xd.transpose(-2, -1)
            zd = torch.rand(0, 0, device=device, dtype=dtype)

            xls, yls, zls = (t.to_sparse(layout=lhs_layout) for t in (xd, yd, zd))
            xrs, yrs, zrs = (t.to_sparse(layout=rhs_layout) for t in (xd, yd, zd))

            for ls, rs, ld, rd in [(xls, yrs, xd, yd), (xls, zrs, xd, zd), (zls, yrs, zd, yd), (zls, zrs, zd, zd)]:
                res_sparse = ls @ rs
                res_dense = ld @ rd
                self.assertEqual(res_sparse.to_dense(), res_dense)

        def test_orthogonal_inputs(lhs_layout, rhs_layout):
            ones = torch.ones(2, 2, device=device, dtype=dtype)
            zeros = torch.zeros(2, 2, device=device, dtype=dtype)
            x = torch.cat((ones, zeros), -1).to_sparse(layout=lhs_layout)
            y = torch.cat((zeros, ones), -2).to_sparse(layout=rhs_layout)
            res = x @ y
            res_expected = torch.zeros(*res.shape, device=device, dtype=dtype, layout=res.layout)
            self.assertEqual(res, res_expected)

        for lhs_layout, rhs_layout in itertools.product([torch.sparse_csr, torch.sparse_csc], repeat=2):
            test_empty_inputs(lhs_layout, rhs_layout)
            test_orthogonal_inputs(lhs_layout, rhs_layout)

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
    @sparse_compressed_nonblock_layouts()
    @skipCUDAIf(
        not _check_cusparse_spgemm_available(),
        "cuSparse Generic API SpGEMM is not available"
    )
    def test_addmm_all_sparse_csr(self, device, dtype, layout):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="all_sparse")

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="all_sparse")

        # Test beta=0, M=nan
        M = torch.full((10, 25), float('nan'), device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, beta=0, layout=layout, mode="all_sparse")

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            _test_addmm_addmv(self, torch.addmm, M, m1, m2, transpose_out=t4, layout=layout, mode="all_sparse")

    @onlyCPU
    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @sparse_compressed_nonblock_layouts()
    def test_addmm_dense_result(self, device, dtype, layout):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="dense_result")

        # Test 0-strided
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="dense_result")

        # Test beta=0, M=nan
        M = torch.full((10, 25), float('nan'), device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, beta=0, layout=layout, mode="dense_result")

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            _test_addmm_addmv(self, torch.addmm, M, m1, m2, transpose_out=t4, layout=layout, mode="dense_result")

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
            expected = fn(y.to_dense(), x.to_dense())
            self.assertEqual(res_sparse_sparse, expected)
            # TODO: While result of mul(dense, csr) is csr, it is not fully compressed.
            # That means it may contain materialized zeros, since the dense argument
            # is converted according to the sparsity pattern of csr. In the future
            # we might require the result to be fully compressed.
            self.assertEqual(res_dense_sparse, expected)
            self.assertEqual(res_sparse_dense, expected)

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

    # TODO: enable hybrid once to_dense supports it
    @parametrize('enable_hybrid', [False])
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half))
    def test_mul_scalar(self, layout, device, dtype, enable_hybrid):
        for sparse in self.generate_simple_inputs(
                layout, device=device, dtype=dtype, index_dtype=torch.int32, enable_hybrid=enable_hybrid):
            for scalar_dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
                # ComplexHalf is experimental
                if dtype is torch.half and scalar_dtype.is_complex:
                    continue

                scalar_t = torch.tensor(2, dtype=scalar_dtype)
                for scalar in (scalar_t, scalar_t.item()):
                    res_out = sparse.mul(scalar)
                    self.assertEqual(res_out, scalar * sparse)

                    res_dense_out = sparse.to_dense().mul(scalar)
                    # BUG: dispatcher ignores mul.Scalar(Tensor, Scalar)
                    # This issues is circumvented in the mul(Tensor, Tensor) kernel.
                    self.assertEqual(res_out, res_dense_out)

                    if dtype == torch.result_type(sparse, scalar):
                        res_in_dense = sparse.to_dense().mul_(scalar)
                        res_in = sparse.clone().mul_(scalar)
                        self.assertEqual(res_in, res_in_dense)
                        self.assertEqual(res_out, res_in)

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
            sparse_args = [S1, S2, S3]
            dense_args = [t.to_dense() for t in sparse_args]
            arg_idx = list(range(len(sparse_args)))
            out_idx = arg_idx + [None]

            for idx1, idx2, idx3 in itertools.product(arg_idx, arg_idx, out_idx):
                s1 = sparse_args[idx1]
                s2 = sparse_args[idx2]
                s3 = None if idx3 is None else sparse_args[idx3]
                d1 = dense_args[idx1]
                d2 = dense_args[idx2]
                d3 = None if idx3 is None else dense_args[idx3]

                expected = torch.add(d1, d2, alpha=alpha, out=d3)
                actual = torch.add(s1, s2, alpha=alpha, out=s3)
                self.assertEqual(actual.crow_indices().dtype, index_dtype)
                self.assertEqual(actual.col_indices().dtype, index_dtype)
                self.assertEqual(actual, expected)
                self.assertEqual(s3, d3)
                if s3 is not None:
                    self.assertEqual(s3.crow_indices().dtype, index_dtype)
                    self.assertEqual(s3.col_indices().dtype, index_dtype)

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
            if not unitriangular:
                triangle_function = torch.triu if upper else torch.tril
            else:
                # Make sure diagonal elements are not materialized.
                # This is to exercise `unitriangular=True` not relying on
                # explicit presence of these indices.
                if upper:
                    def remove_diagonal(t):
                        return t.triu(-1)

                else:
                    def remove_diagonal(t):
                        return t.tril(-1)

                triangle_function = remove_diagonal

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

        mnk = list(itertools.product([2, 5], repeat=3))

        # Add a test case for size 0 a and b tensors
        mnk = mnk + [(5, 5, 0)]

        batch_shapes = [(), (2,), (2, 3)]
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
            c1, a1, b1 = (x.detach().to_dense().requires_grad_(True) for x in [c, a, b])
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

    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 0.01})
    def test_sparse_mm_reduce_sum(self, device, dtype):
        def run_test(m, n, k, nnz, train):
            sparse = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=torch.int64)
            dense = sparse.to_dense()

            mat = torch.randn(k, n, dtype=dtype)
            ref_mat = mat.clone()

            if train:
                sparse.requires_grad_()
                mat.requires_grad_()
                dense.requires_grad_()
                ref_mat.requires_grad_()

            ref_out = torch.mm(dense, ref_mat)
            out = torch.sparse.mm(sparse, mat, 'sum')

            self.assertEqual(out, ref_out)

            if train:
                ref_out.sum().backward()
                out.sum().backward()

                grad_input = sparse.grad
                ref_grad_input = dense.grad
                grad_mat = mat.grad
                ref_grad_mat = ref_mat.grad

                self.assertEqual(grad_input.to_dense(), ref_grad_input)
                self.assertEqual(grad_mat, ref_grad_mat)

        run_test(4, 5, 4, 10, False)
        run_test(4, 4, 4, 16, True)

    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 0.01})
    def test_sparse_mm_reduce(self, device, dtype):
        def run_test(m, n, k, nnz, reduce_type, index_dtype, train):
            csr = self.genSparseCSRTensor((m, n), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            mat = torch.randn(n, k, dtype=dtype)
            ref_mat = mat.clone()
            ref_values = csr.values().clone()

            out_int32 = index_dtype == torch.int32
            coo_indices = torch._convert_indices_from_csr_to_coo(
                csr.crow_indices(),
                csr.col_indices(),
                out_int32=out_int32)
            row, col = coo_indices[0], coo_indices[1]

            def ref(row, col, val, mat):
                out = torch.zeros([m, k], dtype=dtype)
                weight = mat.index_select(0, col)
                src = weight.mul(val.view(-1, 1))
                index = row.view(-1, 1).expand_as(weight)
                index = index.to(dtype=torch.int64)
                # scatter_reduce expect index to be int64
                out.scatter_reduce_(0, index, src, reduce=reduce_type, include_self=False)
                return out

            if train:
                csr.requires_grad_()
                mat.requires_grad_()
                ref_values.requires_grad_()
                ref_mat.requires_grad_()

            ref_out = ref(row, col, ref_values, ref_mat)
            out = torch.sparse.mm(csr, mat, reduce_type)
            self.assertEqual(out, ref_out)

            if train and dtype is not torch.bfloat16:
                ref_out.sum().backward()
                out.sum().backward()

                grad_values = csr.grad.values()
                grad_weight = mat.grad
                ref_grad_values = ref_values.grad
                ref_grad_weight = ref_mat.grad
                self.assertEqual(grad_values, ref_grad_values)
                self.assertEqual(grad_weight, ref_grad_weight)

        for train in [False, True]:
            for index_dtype in [torch.int32, torch.int64]:
                for reduce_type in ["sum", "mean", "amax", "amin"]:
                    # by setting nnz < M, create empty rows
                    run_test(3, 4, 11, 1, reduce_type, index_dtype, train)
                    run_test(3, 4, 11, 6, reduce_type, index_dtype, train)
                    run_test(3, 4, 11, 12, reduce_type, index_dtype, train)
                    # we are doing blocking with 4x vector length in the kernel,
                    # so need to test when K > 4x vector length
                    run_test(4, 7, 33, 13, reduce_type, index_dtype, train)

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

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
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
            # We must skip samples of low dimensionality, we can't covert them to sparsed compressed layouts
            if sample.input.ndim < 2:
                continue
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

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
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
    @all_sparse_compressed_layouts()
    def test_transpose(self, device, dtype, layout):

        def _check_transpose_view(subject, transpose):
            self.assertTrue(transpose.values()._is_view())
            self.assertTrue(transpose._is_view())
            self.assertTrue(transpose._base is subject)

        def _check_layout_invariants(transpose):
            self.assertEqual(transpose.device, torch.device(device))
            compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[transpose.layout]
            compressed_indices, plain_indices = compressed_indices_mth(transpose), plain_indices_mth(transpose)
            torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, transpose.values(),
                                                          transpose.shape, transpose.layout)

        def check_good_transpose(subject, subject_dense, dim0, dim1, expected_layout):
            transpose = subject.transpose(dim0, dim1)
            # correct layout
            self.assertEqual(transpose.layout, expected_layout)
            # transpose must be return a view
            _check_transpose_view(subject, transpose)
            # result uses unsafe construction, so we check invariants
            _check_layout_invariants(transpose)
            self.assertEqual(transpose.to_dense(), subject_dense.transpose(dim0, dim1))

            round_trip = transpose.transpose(dim0, dim1)
            self.assertEqual(round_trip.layout, subject.layout)
            # transpose must be return a view
            _check_transpose_view(subject, round_trip)
            # result uses unsafe construction, so we check invariants
            _check_layout_invariants(round_trip)
            self.assertEqual(round_trip.to_dense(), subject_dense)

        def check_same_dim_transpose(subject, subject_dense, dim):
            transpose = subject.transpose(dim, dim)
            # correct layout
            self.assertEqual(transpose.layout, subject.layout)
            # transpose must be return a view
            _check_transpose_view(subject, transpose)
            # result uses unsafe construction, so we check invariants
            _check_layout_invariants(transpose)
            self.assertEqual(transpose.to_dense(), subject_dense)

        def check_dim_type_mismatch_throws(subject, name0, dim0, name1, dim1):
            mismatch_name = f"{dim0}\\({name0}\\) and {dim1}\\({name1}\\)"
            err = r"transpose\(\): can only transpose dimensions of the same type \(Batch, Sparse, Dense\), got " + mismatch_name

            with self.assertRaisesRegex(RuntimeError, err):
                subject.transpose(dim0, dim1)

        def run_test(shape, nnz, index_type, n_dense, blocksize=()):
            subject = self.genSparseCompressedTensor(shape,
                                                     nnz,
                                                     layout=layout,
                                                     device=device,
                                                     index_dtype=index_type,
                                                     blocksize=blocksize,
                                                     dense_dims=n_dense,
                                                     dtype=dtype)


            sparse0 = len(shape) - n_dense - 1
            sparse1 = sparse0 - 1

            dense0 = sparse0 + 1 if n_dense > 0 else None
            dense1 = dense0 + 1 if n_dense > 1 else None

            n_batch = len(shape) - n_dense - 2
            batch0 = sparse1 - 1 if n_batch > 0 else None
            batch1 = 0 if n_batch > 1 else None

            sparse_dims = (sparse0, sparse1)
            dense_dims = (dense0, dense1)
            batch_dims = (batch0, batch1)

            named0 = [(name, d[0]) for name, d in zip(["Batch", "Sparse", "Dense"], (batch_dims, sparse_dims, dense_dims))]
            named1 = [(name, d[1]) for name, d in zip(["Batch", "Sparse", "Dense"], (batch_dims, sparse_dims, dense_dims))]

            flipped_layout = {
                torch.sparse_csr: torch.sparse_csc,
                torch.sparse_csc: torch.sparse_csr,
                torch.sparse_bsr: torch.sparse_bsc,
                torch.sparse_bsc: torch.sparse_bsr
            }[layout]
            if n_dense > 0:
                # expect all transpose to throw
                for (name0, dim0), (name1, dim1) in itertools.product(named0, named1):
                    msg = r"transpose\(\): hybrid sparse compressed tensors with dense dimensions are not supported"
                    if (dim0 is not None) and (dim1 is not None):
                        with self.assertRaisesRegex(RuntimeError, msg):
                            subject.transpose(dim0, dim1)
            else:
                subject_dense = subject.to_dense()
                for (name0, dim0), (name1, dim1) in itertools.product(named0, named1):
                    if dim0 is not None:
                        check_same_dim_transpose(subject, subject_dense, dim0)

                        if dim1 is not None:
                            if name0 == name1:
                                expected_layout = flipped_layout if name0 == "Sparse" else layout
                                check_good_transpose(subject, subject_dense, dim0, dim1, expected_layout)
                            else:
                                check_dim_type_mismatch_throws(subject, name0, dim0, name1, dim1)

        # batch/sparse, sparse/dense only and full hybrid cases
        shape_ndense = list(itertools.product([(2, 4, 6, 2), (10, 6, 4, 2), (2, 4, 4, 2, 6)], [0, 1, 2]))
        # sparse only cases
        shape_ndense += [[(4, 8), 0], [(2, 2), 0], [(8, 4), 0]]
        for (shape, n_dense), index_dtype in itertools.product(shape_ndense, [torch.int32, torch.int64]):
            n_batch = len(shape) - n_dense - 2
            sparse_shape = shape[n_batch: n_batch + 2]
            if layout in (torch.sparse_bsr, torch.sparse_bsc):
                # for blocked all combinations of 2,1 should be valid blocksizes
                run_test(shape, 0, index_dtype, n_dense, blocksize=(2, 2))
                run_test(shape, max(sparse_shape), index_dtype, n_dense, blocksize=(2, 2))
                run_test(shape, sparse_shape[0] * sparse_shape[1], index_dtype, n_dense, blocksize=(2, 2))
                # repeat the realistic sparseity case with varried block sizes
                run_test(shape, max(sparse_shape), index_dtype, n_dense, blocksize=(2, 1))
                run_test(shape, max(sparse_shape), index_dtype, n_dense, blocksize=(1, 2))
                run_test(shape, max(sparse_shape), index_dtype, n_dense, blocksize=(1, 1))
            else:
                run_test(shape, 0, index_dtype, n_dense)
                run_test(shape, max(sparse_shape), index_dtype, n_dense)
                run_test(shape, sparse_shape[0] * sparse_shape[1], index_dtype, n_dense)

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
        """This test performs a smoke test for covered conversion and verifies
        that an exception is thrown for unsupported conversions.

        TODO: This test covers a subset of
        TestSparseAny.test_to_sparse tests and can be
        eliminated. Keeping the test until the new
        `Tensor.to_sparse(*, layout, blocksize)` has landed.
        """

        allowed_pairwise_layouts_sets = {
            frozenset({torch.sparse_csc}),
            frozenset({torch.sparse_csr}),
            frozenset({torch.sparse_csc, torch.sparse_csr}),
            frozenset({torch.sparse_csc, torch.sparse_bsc}),
            frozenset({torch.sparse_csc, torch.sparse_bsr}),
            frozenset({torch.sparse_csr, torch.sparse_bsc}),
            frozenset({torch.sparse_csr, torch.sparse_bsr}),
            frozenset({torch.sparse_bsc}),
            frozenset({torch.sparse_bsr}),
            frozenset({torch.sparse_bsc, torch.sparse_bsr}),
        }
        block_layouts = (torch.sparse_bsr, torch.sparse_bsc)

        def _to_from_layout(layout_a, layout_b, a):
            expect_error = True
            if {layout_a, layout_b} in allowed_pairwise_layouts_sets:
                expect_error = False

            # BSR -> CSR is not yet supported
            if (layout_a, layout_b) == (torch.sparse_bsr, torch.sparse_csr):
                expect_error = True
            # BSR -> CSC is not yet supported
            if (layout_a, layout_b) == (torch.sparse_bsr, torch.sparse_csc):
                expect_error = True
            # BSC -> CSR is not yet supported
            if (layout_a, layout_b) == (torch.sparse_bsc, torch.sparse_csr):
                expect_error = True
            # BSC -> CSC is not yet supported
            if (layout_a, layout_b) == (torch.sparse_bsc, torch.sparse_csc):
                expect_error = True
            # CSR -> BSR only works for non-batched inputs
            if (layout_a, layout_b) == (torch.sparse_csr, torch.sparse_bsr):
                if a.dim() > 2:
                    expect_error = True
            # CSR -> BSC only works for non-batched inputs
            if (layout_a, layout_b) == (torch.sparse_csr, torch.sparse_bsc):
                if a.dim() > 2:
                    expect_error = True
            # CSC -> BSR only works for non-batched inputs
            if (layout_a, layout_b) == (torch.sparse_csc, torch.sparse_bsr):
                if a.dim() > 2:
                    expect_error = True
            # CSC -> BSC only works for non-batched inputs
            if (layout_a, layout_b) == (torch.sparse_csc, torch.sparse_bsc):
                if a.dim() > 2:
                    expect_error = True

            blocksize_a = (1, 1) if layout_a in {torch.sparse_bsr, torch.sparse_bsc} else None
            blocksize_b = (1, 1) if layout_b in {torch.sparse_bsr, torch.sparse_bsc} else None
            b = a.to_sparse(layout=layout_a, blocksize=blocksize_a)
            if expect_error:
                with self.assertRaises(RuntimeError):
                    b.to_sparse(layout=layout_b, blocksize=blocksize_b)
            else:
                c = b.to_sparse(layout=layout_b, blocksize=blocksize_b)
                self.assertEqual(a.to_dense(), c.to_dense())

                # change of blocksize upon conversion is not yet supported.
                if b.layout in block_layouts:
                    for block_layout in block_layouts:
                        with self.assertRaisesRegex(RuntimeError,
                                                    "conversion from.*to.*with blocksize changed from.*to.*is not supported"):
                            b.to_sparse(layout=block_layout, blocksize=(3, 3))

        batch_dims = [(), (2,), (2, 2), (2, 2, 2)]
        sparse_dims = (6, 12)
        for batch_dim in batch_dims:
            a = make_tensor(batch_dim + sparse_dims, dtype=torch.float, device=device)
            _to_from_layout(from_layout, to_layout, a)

    @skipMeta
    @all_sparse_compressed_layouts()
    @batched_nonbatched()
    @hybrid_nonhybrid()
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_dense_to_from_sparse_compressed(self, device, hybrid, batched, layout):
        """This test tests conversion from dense to/from CSR and CSC
        by comparing to SciPy's implementation.

        Here we test only those conversion combinations that SciPy
        supports to ensure that PyTorch conversions are in the same
        page with SciPy.  Independent from SciPy, all conversion
        combinations are tested in TestSparseAny.test_to_sparse.
        """

        blocked_layouts = (torch.sparse_bsr, torch.sparse_bsc)

        # helpers

        def _check_against_scipy_matrix(pt_matrix, dense, blocksize, **kwargs):
            # scipy has no bsc layout, so we check against the bsr layout of the tranposed dense
            if layout == torch.sparse_bsc:
                sp_matrix = self._construct_sp_matrix(dense.t(), layout=torch.sparse_bsr, blocksize=blocksize[::-1])
            else:
                sp_matrix = self._construct_sp_matrix(dense, layout=layout, blocksize=blocksize)

            compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]

            self.assertEqual(layout, pt_matrix.layout)
            if layout == torch.sparse_bsc:
                self.assertEqual(sp_matrix.shape[::-1], pt_matrix.shape)
            else:
                self.assertEqual(sp_matrix.shape, pt_matrix.shape)

            self.assertEqual(torch.tensor(sp_matrix.indptr, dtype=torch.int64), compressed_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.indices, dtype=torch.int64), plain_indices_mth(pt_matrix))
            if layout == torch.sparse_bsc:
                # we must tranpose the blocks before comparing
                self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values().transpose(-2, -1))
            else:
                self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values())

        def _check_hybrid_matrix(pt_matrix, dense, blocksize, **kwargs):
            # Calculate COO indices for sparse matrix.
            compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
            compressed_indices = compressed_indices_mth(pt_matrix)
            plain_indices = plain_indices_mth(pt_matrix)
            coo_indices = torch._convert_indices_from_csr_to_coo(compressed_indices, plain_indices)
            row_indices, col_indices = {
                torch.sparse_csr: (coo_indices[0, ], coo_indices[1, ]),
                torch.sparse_csc: (coo_indices[1, ], coo_indices[0, ]),
                torch.sparse_bsr: (coo_indices[0, ], coo_indices[1, ]),
                torch.sparse_bsc: (coo_indices[1, ], coo_indices[0, ]),
            }[pt_matrix.layout]

            # If sparse matrix layout blocked, rearrange dense matrix
            # so that the shape past first two dimensions match the
            # shape of sparse matrix values.
            dense_to_check = dense
            if blocksize:
                dense_shape = dense.shape
                dense_to_check_shape = (dense.shape[0] // blocksize[0],
                                        blocksize[0],
                                        dense.shape[1] // blocksize[1],
                                        blocksize[1]) + dense.shape[2:]
                dense_to_check = dense_to_check.reshape(dense_to_check_shape).transpose(1, 2)

            # Verify that non-zero values of the sparse matrix are
            # equal to corresponding values of the dense matrix.
            self.assertEqual(pt_matrix.values(), dense_to_check[row_indices, col_indices])

            # Verify that the remaining elements of the dense matrix
            # are 0, i.e. that dense are sparse matrix are fully
            # equal.
            mask = torch.ones_like(dense_to_check, dtype=torch.bool)
            mask[row_indices, col_indices] = False
            self.assertTrue(torch.all(torch.masked_select(dense_to_check, mask) == 0))

        def _check_batched(pt_tensor, dense, check_batch=None, batch_shape=(), blocksize=(), **kwargs):
            self.assertEqual(layout, pt_tensor.layout)
            self.assertEqual(pt_tensor.shape, dense.shape)
            compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
            for batch_index in np.ndindex(batch_shape):
                pt_matrix = pt_tensor[batch_index]
                dense_matrix = dense[batch_index]
                dense_dim = pt_matrix.dim() - 2
                dense_matrix_pt = dense_matrix.to_sparse(layout=layout,
                                                         blocksize=blocksize or None,
                                                         dense_dim=dense_dim)
                # sanity check, selecting batch of to_<layout> and dense[batch].to_<layout> should give the same result
                self.assertEqual(pt_matrix, dense_matrix_pt)
                check_batch(pt_matrix, dense_matrix, blocksize, **kwargs)

        def _generate_subject(sparse_shape, batch_shape, hybrid_shape):
            shape = batch_shape + sparse_shape + hybrid_shape
            n_batch_dim = len(batch_shape)
            n_hybrid_dim = len(hybrid_shape)
            # generate a dense tensor
            dense = make_tensor(shape, dtype=torch.float, device=device)

            # introduce some sparsty, mask is sparse shape, element applies to entire dense sub-tensor (hybrid) and is
            # applied to each batch
            mask = make_tensor(sparse_shape, dtype=torch.bool, device=device)
            # manually expand to match hybrid shape
            if hybrid:
                mask = mask.view(sparse_shape + tuple(1 for _ in range(n_hybrid_dim)))
                mask = mask.expand(sparse_shape + hybrid_shape)

            # mask will broadcast over the batch dims if present

            return dense * mask

        # note: order is important here, the hybrid-ness decides the inner content check which is used to build the
        # batched checker (if needed)
        check_content = _check_against_scipy_matrix
        if hybrid:
            check_content = _check_hybrid_matrix
        if batched:
            check_content = functools.partial(_check_batched, check_batch=check_content)

        sparse_sizes = [(6, 10), (0, 10), (6, 0), (0, 0)]
        blocksizes = [(2, 2), (1, 1), (1, 2)] if layout in blocked_layouts else [()]
        batch_sizes = [(3,), (1, 3), (2, 1, 3)] if batched else [()]
        hybrid_sizes = [(4, ), (2, 2)] if hybrid else [()]

        # general cases, always run
        for sparse_shape, blocksize, batch_shape, hybrid_shape in itertools.product(
                sparse_sizes, blocksizes, batch_sizes, hybrid_sizes):
            dense = _generate_subject(sparse_shape, batch_shape, hybrid_shape)
            sparse = dense.to_sparse(layout=layout, blocksize=blocksize or None, dense_dim=len(hybrid_shape))
            check_content(sparse, dense, blocksize=blocksize, batch_shape=batch_shape, hybrid_shape=hybrid_shape)
            dense_back = sparse.to_dense()
            self.assertEqual(dense, dense_back)

        # special cases for batched tensors
        if batched:
            # batched sparse tensors need only have the same number of non-zeros in each batch not nessesarily the
            # same sparsity pattern in each batch
            sparse_shape = sparse_sizes[0]
            hybrid_shape = hybrid_sizes[0]
            batch_shape = batch_sizes[0]
            shape = batch_shape + sparse_shape + hybrid_shape
            dense = make_tensor(shape, dtype=torch.float, device=device)
            blocksize = blocksizes[0]
            # number of elements/blocks in each batch (total not nnz)
            batch_mask_shape = sparse_shape
            if layout in blocked_layouts:
                # if we are blocked the mask is genereated for the block valued elemetns
                batch_mask_shape = sparse_shape[0] // blocksize[0], sparse_shape[1] // blocksize[1]

            # random bool vector w/ length equal to max possible nnz for the sparse_shape
            mask_source = make_tensor(batch_mask_shape, dtype=torch.bool, device=device).flatten()
            n_batch = functools.reduce(lambda x, y: x * y, batch_shape, 1)

            # stack random permutations of the source for each batch
            mask = torch.stack([mask_source[torch.randperm(mask_source.numel())]
                               for _ in range(n_batch)], dim=0).reshape(batch_shape + batch_mask_shape)
            if layout in blocked_layouts:
                # for blocked we need to do a bit of extra work to expand the mask from blocked-space to element-space
                mask_shape = mask.shape
                mask = mask.view(mask_shape + (1, 1))
                mask = mask.expand(mask_shape + blocksize)
                mask = mask.transpose(-3, -2)
                mask = mask.flatten(-4, -3).flatten(-2, -1)
            mask_shape = mask.shape
            mask = mask.view(mask_shape + (1,) * len(hybrid_shape))
            mask = mask.expand(mask_shape + hybrid_shape)
            dense = dense * mask
            sparse = dense.to_sparse(layout=layout, blocksize=blocksize or None, dense_dim=len(hybrid_shape))
            check_content(sparse, dense, blocksize=blocksize, batch_shape=batch_shape, hybrid_shape=hybrid_shape)

            dense_back = sparse.to_dense()
            self.assertEqual(dense, dense_back)

            # if batches have different nnz we expect the conversion to throw
            mask_0 = mask[0]
            mask_1 = mask[0].clone().fill_(True)
            mask_2 = mask[0].clone().fill_(False)
            mask_true = mask_source.clone().fill_(True)
            mask_false = mask_source.clone().fill_(False)
            mask = torch.stack([(mask_0, mask_1, mask_2)[i % 3] for i in range(n_batch)], dim=0).reshape(batch_shape + mask_0.shape)
            dense = make_tensor(shape, dtype=torch.float, device=device)
            dense = dense * mask
            msg = "Expect the same number of specified elements per batch."
            with self.assertRaisesRegex(RuntimeError, msg):
                dense.to_sparse(layout=layout, blocksize=blocksize or None)

            # Should throw if there is a zero in the batch size
            dense = make_tensor((0,) + shape, dtype=torch.float, device=device)
            layout_code = str(layout).split("_")[-1]
            msg = f"to_sparse_{layout_code}: Expected product of batch dimensions to be non-zero."
            with self.assertRaisesRegex(RuntimeError, msg):
                dense.to_sparse(layout=layout, blocksize=blocksize or None)

    @skipMeta
    @all_sparse_compressed_layouts()
    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_sparse_to_sparse_compressed(self, device, dtype, coalesced, layout):
        """
        This test tests conversion from COO to CSR and CSC and CSC to CSR and CSC
        by comparing to SciPy's implementation.

        Here we test only those conversion combinations that SciPy
        supports to ensure that PyTorch conversions are in the same
        page with SciPy.  Independent from SciPy, all conversion
        combinations are tested in TestSparseAny.test_to_sparse.
        """
        if layout is torch.sparse_bsc:
            # TODO: Remove this once support has been enabled
            self.skipTest('NOT IMPL')
        if layout is torch.sparse_bsr:
            # TODO: Remove this once support has been enabled
            self.skipTest('NOT IMPL')

        for shape in [(0, 10), (6, 0), (6, 10), (0, 0)]:
            sparse_dim = 2
            nnz = shape[0] * shape[1] // 2
            sparse, _, _ = self.genSparseTensor(shape, sparse_dim, nnz, coalesced, device, dtype)
            sp_matrix = self._construct_sp_matrix(sparse, layout)
            pt_matrix = sparse.to_sparse(layout=layout)

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
            pt_matrix = sparse_csc.to_sparse(layout=layout)

            self.assertEqual(layout, pt_matrix.layout)
            self.assertEqual(sp_matrix.shape, pt_matrix.shape)
            self.assertEqual(torch.tensor(sp_matrix.indptr, dtype=torch.int64), compressed_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.indices, dtype=torch.int64), plain_indices_mth(pt_matrix))
            self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values())


def skipIfNoTriton(cls):
    from torch._inductor.utils import has_triton

    # no-op if triton is present
    if has_triton():
        return cls
    else:

        @functools.wraps(cls, updated=())
        class skipped_cls(cls):
            def setUp(self):
                self.skipTest("Triton is not available.")

        return skipped_cls

@skipIfNoTriton
class TestSparseCompressedTritonKernels(TestCase):

    def _to_block_triangular_inplace(self, d, row_block, col_block):
        """
        This function modifies `d` to become (upper/lower) block-triangular in-place.
        It is assumed that `d.shape[-2]` is divisible by `row_block` and
        `d.shape[-1]` is divisible by `col_block`.
        """

        from torch.sparse._triton_ops import tile_to_blocksize

        m, n = d.shape[-2:]
        d_tiled = tile_to_blocksize(d, (row_block, col_block))
        d_tiled = d_tiled.moveaxis(-4, -1).moveaxis(-4, -1)
        if m // row_block > n // col_block:
            d_tiled.tril_()
        else:
            d_tiled.triu_()

        return d

    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.half, torch.bfloat16, torch.float)
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    def test_triton_bsr_softmax(self, device, dtype):
        from functools import partial
        from torch.sparse._triton_ops import bsr_softmax

        tensor = partial(make_tensor, device=device, dtype=dtype, low=1.0, high=3.0)

        # NOTE: batch dims with zero sizes are not supported in `to_sparse_bsr`.
        batches = [(), (2,), (2, 2)]
        size = [6, 12, 0]
        block_size = [2, 3]

        # General correctness
        for row_block, col_block, b, m, n in itertools.product(block_size, block_size, batches, size, size):
            input = tensor(b + (m, n))
            input.diagonal(dim1=-2, dim2=-1).fill_(m * n)
            input = self._to_block_triangular_inplace(input, row_block, col_block)

            bsr = input.to_sparse_bsr((row_block, col_block))
            coo = input.to_sparse().to(torch.float)

            res_tri = bsr_softmax(bsr)
            res_coo = torch.sparse.softmax(coo, -1)
            self.assertEqual(res_tri, res_coo.to(input.dtype))

        # Test long rows which exceed Triton's max numel limit set to 2 ** 17
        input = tensor(b + (1, 150000))
        bsr = input.to_sparse_bsr(1)
        self.assertEqual(input.softmax(-1), bsr_softmax(bsr))

    @parametrize("block_size", [16, 32, 64])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.half, torch.bfloat16, torch.float)
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    @unittest.skipIf((not TEST_WITH_TORCHINDUCTOR) or (IS_FBCODE and IS_REMOTE_GPU) or torch._running_with_deploy(),
                     "Skipped for deploy and internal with remote GPUs")
    def test_triton_bsr_dense_bmm(self, device, dtype, index_dtype, block_size):
        from functools import partial
        from torch.sparse._triton_ops import bsr_dense_mm

        def kernel_impl(*args, **kwargs):
            return bsr_dense_mm(*args, skip_checks=True, **kwargs)

        kernel = torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_mm_out",
            "_triton_bsr_dense_mm_out(Tensor bsr, Tensor dense, *, Tensor(a!) out) -> Tensor(a!)",
            kernel_impl,
            "SparseCsrCUDA"
        )

        # kernel != kernel_impl means dispatch was already registered.
        # This is exactly what we need!
        self.assertTrue(kernel is not kernel_impl)

        # Note that each value in a non-zero block is in range block_size * [low^2, high^2).
        tensor = partial(make_tensor, device=device, dtype=dtype, low=0.5, high=1.5)

        # NOTE: batch dims with zero sizes are not supported in `to_sparse_bsr`.
        batches = [(), (2,), (2, 2)]
        size = [128, 256, 0]

        # Whether to make inputs orthogonal so that the product is zero
        make_orthogonal = [True, False]

        for bd, bs, m, n, k, is_ortho in itertools.product(batches, batches, size, size, size, make_orthogonal):
            bsr = tensor(bs + (m, k))
            # NOTE: do not get confused, it will be transposed
            dense = tensor(bd + (n, k))

            if is_ortho:
                bsr = torch.cat((bsr, torch.zeros_like(bsr)), dim=-1)
                dense = torch.cat((torch.zeros_like(dense), dense), dim=-1)

            bsr = bsr.to_sparse_bsr(block_size)

            if bsr.dim() == 2 and dtype != torch.float:
                # Test against linear to check dispatch
                # which takes place for torch.half and torch.bfloat16.
                res_dense = torch.nn.functional.linear(dense, bsr.to_dense())
                res_tri_out = torch.empty_like(res_dense)
                res_tri = torch.nn.functional.linear(dense, bsr, out=res_tri_out)

                # Check dispatch worked with non-trivial outputs
                if m > 0 and n > 0 and k > 0:
                    self.assertTrue(kernel.kernel_invoked)
                    kernel.kernel_invoked = False
            else:
                # Otherwise check correctness against bmm
                # since nn.linear does not support bsr.dim() > 2.
                res_dense = bsr.to_dense() @ dense.transpose(-2, -1)
                res_tri_out = torch.empty_like(res_dense)
                res_tri = kernel(bsr, dense.transpose(-2, -1), out=res_tri_out)

            self.assertTrue(res_tri is res_tri_out)
            self.assertEqual(res_tri, res_dense)

            res_dense = bsr.to_dense() @ dense.transpose(-2, -1)
            # check whether bsr_dense_mm handles different grid sizes
            # None means max possible grid size which is CUDA-dependent.
            grid_size = (None, 2, 4)
            grid_gen = itertools.product(grid_size, repeat=3)
            for grid in grid_gen:
                res_tri = torch.sparse._triton_ops.bsr_dense_mm(
                    bsr,
                    dense.transpose(-2, -1),
                    max_grid=grid,
                )
                self.assertEqual(res_tri, res_dense)

    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.half)
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU or torch._running_with_deploy(),
                     "Skipped for deploy and internal with remote GPUs")
    def test_triton_bsr_dense_bmm_error_messages(self, device, dtype):
        from torch.sparse._triton_ops import bsr_dense_mm

        rhs = torch.rand(32, 32, dtype=dtype, device=device)
        lhs = rhs.to_sparse_bsr(16)
        with self.assertRaisesRegex(ValueError, "only BSR sparse format is supported"):
            bsr_dense_mm(lhs.to_sparse_bsc(16), rhs)
        with self.assertRaisesRegex(ValueError, "on the same GPU device"):
            bsr_dense_mm(lhs, rhs.cpu())
        if torch.cuda.device_count() > 1:
            with self.assertRaisesRegex(ValueError, "on the same GPU device"):
                bsr_dense_mm(lhs.to("cuda:0"), rhs.to("cuda:1"))
        with self.assertRaisesRegex(ValueError, "all inputs are expected to be of the same dtype"):
            bsr_dense_mm(lhs, rhs.to(torch.float))
        with self.assertRaisesRegex(ValueError, r"and one of \(half, bfloat16, float32\)"):
            bsr_dense_mm(lhs.to(torch.double), rhs.to(torch.double))
        with self.assertRaisesRegex(ValueError, "all inputs involved in the matrix product are expected to be at least 2D"):
            bsr_dense_mm(lhs, torch.rand(1, dtype=dtype, device=device))
        with self.assertRaisesRegex(ValueError,
                                    "sizes involved in the matrix product are not compatible for matrix multiplication"):
            bsr_dense_mm(lhs, torch.rand(1, 1, dtype=dtype, device=device))
        with self.assertRaisesRegex(ValueError,
                                    r"dense.size\(-1\) == 15 should be divisible by blocksize\[0\] == 16"):
            bsr_dense_mm(lhs, torch.rand(32, 15, dtype=dtype, device=device))
        # Blocksizes check
        for blocksize in (15, 30):
            n = blocksize * 2
            rhs = torch.rand(n, n, dtype=dtype, device=device)
            lhs = rhs.to_sparse_bsr(blocksize)
            with self.assertRaisesRegex(ValueError, "should be at least 16 and a power of 2"):
                bsr_dense_mm(lhs, rhs)
        # out check
        rhs = torch.rand(2, 32, 32, dtype=dtype, device=device)
        lhs = rhs.to_sparse_bsr(16)
        with self.assertRaisesRegex(ValueError, r"`out` argument has wrong shape"):
            out = torch.rand(2, 30, 30, dtype=dtype, device=device)
            bsr_dense_mm(lhs, rhs, out=out)
        with self.assertRaisesRegex(ValueError, r"only row-major/col-major `out`"):
            out = torch.rand(32, 32, 2, dtype=dtype, device=device).transpose(0, -1)
            bsr_dense_mm(lhs, rhs, out=out)

    @parametrize("block_size", [16, 32, 64])
    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.half, torch.bfloat16, torch.float)
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    @precisionOverride({torch.float16: 1e-3})
    def test_triton_scaled_dot_product_attention(self, device, dtype, block_size):
        from functools import partial
        from torch.sparse._triton_ops import _scaled_dot_product_attention

        # Note that each value in a non-zero block is in range block_size * [low^2, high^2).
        tensor = partial(make_tensor, device=device, dtype=dtype, low=0.3, high=1.2)

        def broadcast_input(*ts):
            batch_dims = torch.broadcast_shapes(*(t.shape[:-2] for t in ts))
            yield from (torch.broadcast_to(t, batch_dims + t.shape[-2:]) for t in ts)

        # NOTE: batch dims with zero sizes are not supported in `to_sparse_bsr`.
        batches = [(), (2,), (2, 2)]
        size = [128, 256, 0]

        for bam, bq, bk, bv, m, n, k in itertools.product(batches, batches, batches, batches, size, size, size):
            query = tensor(bq + (m, k))
            key = tensor(bk + (n, k))
            value = tensor(bv + (n, k))

            # We make attn_mask block lower/upper triangular so that BSR and Strided
            # function variants are directly comparable.
            attn_mask = torch.ones(bam + (m, n), device=device, dtype=torch.bool)
            attn_mask = self._to_block_triangular_inplace(attn_mask, block_size, block_size)
            attn_mask_bsr = attn_mask.to_sparse_bsr(block_size)

            # NOTE: only boolean mask is directly compatible with the Strided version
            # without any pre-/post-processing. Hence we test against a boolean mask.
            for scale in (None, 1. / 16):
                if scale is None and query.size(-1) == 0:
                    scale = 1
                expected = torch.nn.functional.scaled_dot_product_attention(
                    *broadcast_input(query, key, value, attn_mask), scale=scale
                )

                for mask_dtype in (torch.bool, dtype):
                    res = _scaled_dot_product_attention(query, key, value, attn_mask_bsr.to(mask_dtype), scale=scale)
                    self.assertEqual(res, expected)


    @parametrize("block_size", [16, 32, 64])
    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.half, torch.bfloat16, torch.float)
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    def test_triton_sampled_addmm(self, device, dtype, block_size):
        from functools import partial
        from torch.sparse._triton_ops import sampled_addmm, broadcast_batch_dims_bsr

        # Note that each value in a non-zero block is in range block_size * [low^2, high^2).
        tensor = partial(make_tensor, device=device, dtype=dtype, low=0.3, high=1.2)

        # NOTE: batch dims with zero sizes are not supported in `to_sparse_bsr`.
        batches = [(), (2,), (2, 2)]
        size = [128, 256, 0]

        delta_k = (-3,)
        for bi, bm1, bm2, m, n, k, dk in itertools.product(batches, batches, batches, size, size, size, delta_k):
            # Test not powers of 2 ks as well.
            k = max(0, k + dk)
            # Non-trivial sparsity pattern.
            # Plus with tril inputs the result is also tril,
            # so we can compare BSR and CSR implementations.
            input = tensor(bi + (m, n)).tril_()
            bsr = input.to_sparse_bsr(block_size)
            mat1 = tensor(bm1 + (m, k)).tril_()
            mat2 = tensor(bm2 + (k, n)).tril_()

            batch_dim = torch.broadcast_shapes(input.shape[:-2], mat1.shape[:-2], mat2.shape[:-2])

            csr = input.broadcast_to(batch_dim + input.shape[-2:]).to_sparse_csr().to(torch.float)
            mat1csr = mat1.broadcast_to(batch_dim + mat1.shape[-2:]).to(torch.float)
            mat2csr = mat2.broadcast_to(batch_dim + mat2.shape[-2:]).to(torch.float)

            input_broadcasted_clone = broadcast_batch_dims_bsr(
                "test_triton_sampled_addmm",
                bsr, mat1, mat2
            ).clone()
            input_broadcasted_clone = torch.sparse_compressed_tensor(
                input_broadcasted_clone.crow_indices(),
                input_broadcasted_clone.col_indices(),
                # For testing `out=` let's make values to have "weird" strides
                # so that if the kernel modifies values to it's needs, the result
                # is being compied into out.values.
                input_broadcasted_clone.values().transpose(-3, -2).contiguous().transpose(-3, -2),
                layout=input_broadcasted_clone.layout,
                size=input_broadcasted_clone.shape
            )

            scalars = (0.0, 2.0)
            for alpha, beta, out in itertools.product(scalars, scalars, (None, input_broadcasted_clone)):
                res_tri = sampled_addmm(bsr, mat1, mat2, alpha=alpha, beta=beta, out=out)
                if out is not None:
                    self.assertTrue(res_tri is out)

                batch_broadcasted_shape = torch.broadcast_shapes(*(t.shape[:-2] for t in (input, mat1, mat2)))
                self.assertTrue(res_tri.shape == batch_broadcasted_shape + (m, n))

                res_csr = torch.sparse.sampled_addmm(csr, mat1csr, mat2csr, alpha=alpha, beta=beta).to(input.dtype)
                self.assertEqual(res_tri.to_dense(), res_csr.to_dense())

                # Check different grid sizes to make sure that input slicing works
                # if this input is larger than the grid.
                grid_size = (3, None)
                grid_gen = itertools.product(grid_size, repeat=2)
                for grid in grid_gen:
                    res_tri_grid = sampled_addmm(bsr, mat1, mat2, alpha=alpha, beta=beta, max_grid=grid)
                    self.assertEqual(res_tri, res_tri_grid)


# e.g., TestSparseCSRCPU and TestSparseCSRCUDA
instantiate_device_type_tests(TestSparseCSR, globals())
instantiate_device_type_tests(TestSparseCompressed, globals())
instantiate_device_type_tests(TestSparseCompressedTritonKernels, globals())

if __name__ == '__main__':
    run_tests()
