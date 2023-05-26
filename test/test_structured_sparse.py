# Owner(s): ["module: sparse"]

import torch
import random
import itertools
import unittest
import functools
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC
from torch.testing._internal.common_utils import \
    (TEST_WITH_ROCM, TEST_SCIPY, TEST_NUMPY, TEST_MKL, IS_WINDOWS, TestCase, run_tests,
     load_tests, coalescedonoff, parametrize, subtest, skipIfTorchDynamo, skipIfRocm, IS_FBCODE, IS_REMOTE_GPU)
from torch.testing._internal.common_device_type import \
    (ops, instantiate_device_type_tests, dtypes, OpDTypes, dtypesIfCUDA, onlyCPU, onlyCUDA, skipCUDAIfNoSparseGeneric,
     precisionOverride, skipMeta, skipCUDAIf, skipCUDAIfRocm, skipCPUIfNoMklSparse, skipCUDAIfRocmVersionLessThan)
from torch.testing._internal.common_methods_invocations import \
    (op_db, sparse_csr_unary_ufuncs, ReductionOpInfo)
from torch.testing._internal.common_cuda import _get_torch_cuda_version, TEST_CUDA
from torch.testing._internal.common_dtype import (
    floating_types, all_types_and_complex_and, floating_and_complex_types, floating_types_and,
    all_types_and_complex, floating_and_complex_types_and
)
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse
from test_sparse import CUSPARSE_SPMM_COMPLEX128_SUPPORTED


class TestSemiStructuredSparse(TestCase):

    def test_csr_stride(self):
        print("ASDF")
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
        msg = "selecting sparse dimensions is not implemented for batched sparse compressed tensors."
        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-2, 0)

        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-1, 0)


instantiate_device_type_tests(TestSemiStructuredSparse, globals())

if __name__ == '__main__':
    run_tests()
