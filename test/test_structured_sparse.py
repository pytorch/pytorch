# Owner(s): ["module: sparse"]

import torch
from torch import nn
import random
import itertools
import unittest
import functools
from torch.sparse import to_semi_structured_sparse, SemiStructuredSparseTensor 
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

CUSPARSELT_SUPPORTED_DTYPES = [torch.float16]


def gen_two_four_sparse_mask(r, c, dtype=torch.float16, device="cuda"):
    def random_mask_choice(i=None):
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
        return choices[random.randint(0, len(choices) - 1) if i is None else i]

    mask_entries = [random_mask_choice() for i in range(r * c // 4)]
    return (
        torch.tensor(mask_entries, dtype=dtype, device=device).view(r, c).contiguous()
    )


def gen_n_m_sparse_mask(r, c, n, m, dtype, device):
    pass


"""
One assumption we make about Tensors are that if they are contiguous, if they are not then they are transposed. 
"""

class TestSemiStructuredSparse(TestCase):
    """


    """

    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_to_semi_structured_sparse(self, dtype):
        """
        test to_semi_
        """

        a = gen_two_four_sparse_mask(128, 128, dtype=dtype)
        a_sparse = to_semi_structured_sparse(a)

        assert a.shape == a_sparse.shape
        assert a.device == a_sparse.device
        assert a.dtype == a_sparse.dtype

        assert isinstance(a, torch.Tensor)
        assert isinstance(a_sparse, SemiStructuredSparseTensor)


    @parametrize('contiguous_output', [subtest(False, name='_'), subtest(True, name='contiguous_output')])
    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_mm_sparse_first(self, dtype, contiguous_output, backend="cusparselt"):
        """

        Test cusparselt matmul
        """

        A = gen_two_four_sparse_mask(64, 64, dtype=dtype)
        A_sparse = to_semi_structured_sparse(A, backend=backend)
        B = torch.rand((64, 64), device=A_sparse.device).to(dtype)

        sparse_result = torch.mm(A_sparse, B)

        dense_result = torch.mm(A, B)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-5, atol=1e-5)

    @parametrize('contiguous_output', [subtest(False, name='_'), subtest(True, name='contiguous_output')])
    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_mm_sparse_second(self, dtype, contiguous_output, backend="cusparselt"):
        """

        Test cusparselt matmul
        """

        A = gen_two_four_sparse_mask(64, 64, dtype=dtype)
        A_sparse = to_semi_structured_sparse(A, backend=backend, transposed=True)
        B = torch.rand((64, 64), device=A_sparse.device).to(dtype)

        sparse_result = torch.mm(B, A_sparse)

        """
        B A_sparse = (A'B')'
        """

        dense_result = torch.mm(B, A)
        print(sparse_result, dense_result)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-5, atol=1e-5)

    @parametrize('contiguous_output', [subtest(False, name='_'), subtest(True, name='contiguous_output')])
    def test_linear(self, contiguous_output):
        SemiStructuredSparseTensor.contiguous_output = contiguous_output
        input = torch.rand(64, 64, device='cuda').half()
        model = nn.Linear(64, 64).cuda().half()
        m, n = model.weight.shape
        model.weight = nn.Parameter(gen_two_four_sparse_mask(m, n))

        dense_result = model(input)
        model.weight = nn.Parameter(to_semi_structured_sparse(model.weight, backend="cusparselt"))
        sparse_result = model(input)
        assert torch.allclose(dense_result, sparse_result, rtol=1e-5, atol=1e-5)

    def test_kept_elements(self):
        A = gen_two_four_sparse_mask(64, 64)
        A_sparse = to_semi_structured_sparse(A)
        assert A_sparse.kept_elements.shape == (64, 32)


    def test_metadata(self):
        A = gen_two_four_sparse_mask(64, 64)
        A_sparse = to_semi_structured_sparse(A)
        assert A_sparse.metadata.shape == (64, 4)



instantiate_device_type_tests(TestSemiStructuredSparse, globals(), only_for="cuda")

if __name__ == '__main__':
    run_tests()
