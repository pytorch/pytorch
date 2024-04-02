# Owner(s): ["module: sparse"]
#
# Test to ensure sparsity information propagates properly into traced graph.
#

import torch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# Common sparse data dtypes currently supported in torch.sparse.
SPARSE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]

# Common sparse index dtypes currently supported in torch.sparse.
SPARSE_ITYPES = [
    torch.int32,
    torch.int64,
]

# Sparse layouts currently supported in torch.sparse.
SPARSE_LAYOUTS = [
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
]


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_copy(self, dtype, itype, layout):
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            # Invoke the dynamo clone input method directly.
            sparse_copy = torch._dynamo.utils.clone_input(sparse_input)
            # Make sure copy is successful.
            self.assertEqual(sparse_input.layout, sparse_copy.layout)
            self.assertEqual(sparse_input.shape, sparse_copy.shape)

    # TODO: actual trace graph propagation tests


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()
