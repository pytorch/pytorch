# Owner(s): ["module: sparse"]
import random

import torch
from torch import nn

from torch.sparse.semi_structured import (
    to_sparse_semi_structured,
    SparseSemiStructuredTensor,
)

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    parametrize,
    subtest,
)

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_dtype import (
    all_types_and_complex
)
CUSPARSELT_SUPPORTED_DTYPES = [torch.int8, torch.float16, torch.bfloat16, torch.float32]

def rand_sparse_semi_structured_mask(r, c, dtype=torch.float16, device="cuda", choice=None):
    """
    This function returns a 1:2 sparse matrix of size (r, c).
    Note that this means this matrix will also be 2:4 and 4:8 sparse as well.
    """

    choices = [[0, 1], [1, 0]]
    mask_entries = [choice or random.choice(choices) for i in range(r * c // 2)]

    return (
        torch.tensor(mask_entries, dtype=dtype, device=device)
        .reshape(r, c)
        .contiguous()
    )

class TestSparseSemiStructured(TestCase):

    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_to_sparse_semi_structured(self, dtype):
        a = rand_sparse_semi_structured_mask(128, 128, dtype=dtype)
        a_sparse = to_sparse_semi_structured(a)

        assert a.shape == a_sparse.shape
        assert a.device == a_sparse.device
        assert a.dtype == a_sparse.dtype

        assert isinstance(a, torch.Tensor)
        assert isinstance(a_sparse, SparseSemiStructuredTensor)

    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_mm_sparse_first_NT_T(self, dtype, device):
        A = rand_sparse_semi_structured_mask(64, 64, dtype=dtype)
        A_sparse = to_sparse_semi_structured(A)

        B = torch.rand((64, 64), device=A_sparse.device).to(dtype)

        sparse_result = torch.mm(A_sparse, B.T)

        # Currently we don't support int matmul on GPU, so evaluate on CPU and copy over
        if dtype is torch.int8:
            dense_result = torch.mm(A.cpu(), B.T.cpu()).to(dtype).to(device)
        else:
            dense_result = torch.mm(A, B.T)

        correct = torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

        if not correct:
            print(dense_result)
            print(sparse_result)

            print(A_sparse)

        assert correct

    @parametrize("inference_mode", [subtest(False), subtest(True)])
    @parametrize("contiguous_output", [subtest(False), subtest(True)])
    def test_linear(self, inference_mode, contiguous_output, device):
        SparseSemiStructuredTensor._fuse_transpose = contiguous_output

        input = torch.rand(64, 64, device=device).half()
        model = nn.Linear(64, 64).to(device).half()
        m, n = model.weight.shape
        model.weight = nn.Parameter(rand_sparse_semi_structured_mask(m, n))

        dense_result = model(input)
        model.weight = nn.Parameter(to_sparse_semi_structured(model.weight))

        if inference_mode:
            with torch.inference_mode():
                sparse_result = model(input)
        else:
            sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-5, atol=1e-5)

    def test_values(self):
        A = rand_sparse_semi_structured_mask(64, 64)
        A_sparse = to_sparse_semi_structured(A)
        assert A_sparse.values().shape == (64, 32)
        assert (A_sparse.values() == 1).all()

    def test_indices(self):
        A = rand_sparse_semi_structured_mask(64, 64)
        A_sparse = to_sparse_semi_structured(A)
        assert A_sparse.indices().shape == (64, 4)

    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_unsupported_shape(self, dtype, device):
        A = rand_sparse_semi_structured_mask(4, 4, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.shape"):
            A_sparse = to_sparse_semi_structured(A)

    @dtypes(*all_types_and_complex())
    def test_unsupported_dtype(self, dtype, device):
        A = rand_sparse_semi_structured_mask(64, 64, dtype=dtype, device=device)

        if dtype not in CUSPARSELT_SUPPORTED_DTYPES:
            with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dtype"):
                A_sparse = to_sparse_semi_structured(A)
        else:
            A_sparse = to_sparse_semi_structured(A)

    def test_unsupported_dim(self, device):
        A = torch.rand(64, 64, 64, device=device, dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "Error original_tensor.dim"):
            A_sparse = to_sparse_semi_structured(A)




instantiate_device_type_tests(TestSparseSemiStructured, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
