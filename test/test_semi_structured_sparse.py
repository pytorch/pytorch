# Owner(s): ["module: sparse"]

import torch
from torch import nn

from torch.sparse import SemiStructuredSparseTensor
from torch.sparse.semi_structured import (
    to_semi_structured_sparse,
    gen_semi_structured_sparse_mask,
)

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    load_tests,
    coalescedonoff,
    parametrize,
    subtest,
    skipIfTorchDynamo,
    skipIfRocm,
    IS_FBCODE,
    IS_REMOTE_GPU,
)

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import (
    floating_types,
    all_types_and_complex_and,
    floating_and_complex_types,
    floating_types_and,
    all_types_and_complex,
    floating_and_complex_types_and,
)

CUSPARSELT_SUPPORTED_DTYPES = [torch.int8, torch.float16, torch.bfloat16, torch.float32]


class TestSemiStructuredSparse(TestCase):
    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_to_semi_structured_sparse(self, dtype):
        a = gen_semi_structured_sparse_mask(128, 128, dtype=dtype)
        a_sparse = to_semi_structured_sparse(a)

        assert a.shape == a_sparse.shape
        assert a.device == a_sparse.device
        assert a.dtype == a_sparse.dtype

        assert isinstance(a, torch.Tensor)
        assert isinstance(a_sparse, SemiStructuredSparseTensor)

    @dtypes(*CUSPARSELT_SUPPORTED_DTYPES)
    def test_mm_sparse_first_NT_T(self, dtype, device):
        A = gen_semi_structured_sparse_mask(64, 64, dtype=dtype)
        A_sparse = to_semi_structured_sparse(A)

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
        SemiStructuredSparseTensor.contiguous_output = contiguous_output

        input = torch.rand(64, 64, device="cuda").half()
        model = nn.Linear(64, 64).cuda().half()
        m, n = model.weight.shape
        model.weight = nn.Parameter(gen_semi_structured_sparse_mask(m, n))

        dense_result = model(input)
        model.weight = nn.Parameter(to_semi_structured_sparse(model.weight))

        if inference_mode:
            with torch.inference_mode():
                sparse_result = model(input)
        else:
            sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-5, atol=1e-5)

    def test_kept_elements(self):
        A = gen_semi_structured_sparse_mask(64, 64)
        A_sparse = to_semi_structured_sparse(A)
        assert A_sparse.kept_elements.shape == (64, 32)

    def test_metadata(self):
        A = gen_semi_structured_sparse_mask(64, 64)
        A_sparse = to_semi_structured_sparse(A)
        assert A_sparse.metadata.shape == (64, 4)


instantiate_device_type_tests(TestSemiStructuredSparse, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
