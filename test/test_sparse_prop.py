# Owner(s): ["module: sparse"]
#
# Test to ensure sparsity information propagates properly into traced graph.
#

from typing import Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor

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


# Ensures we can extract sparsity information from stored meta data.
def extract_sparse_tensor_metadata(
    t: torch.Tensor,
) -> Tuple[int, int, int, Optional[Tuple[int, int]], Optional[torch.dtype]]:
    batch_dim = t.ndim - t.dense_dim() - t.sparse_dim()
    # Set block size.
    if t.layout is torch.sparse_bsr or t.layout is torch.sparse_bsc:
        blocksize = t.values().shape[batch_dim + 1 : batch_dim + 3]
    else:
        blocksize = None
    # Set index type.
    if t.layout is torch.sparse_coo:
        idx_dtype = t._indices().dtype  # supports uncoalesced COO tensors
    elif t.layout is torch.sparse_csr or t.layout is torch.sparse_bsr:
        idx_dtype = t.col_indices().dtype
    else:
        idx_dtype = t.row_indices().dtype
    # Return sparse metadata.
    return (batch_dim, t.sparse_dim(), t.dense_dim(), blocksize, idx_dtype)


class SumNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sum()


class EltwiseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(2 * torch.sin(-x))


class TestSparseProp(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_sumnet(self, dtype, itype, layout):
        # TODO: support more cases
        if layout != torch.sparse_coo:
            self.skipTest("layout support not yet implemented")
        if layout == torch.sparse_coo and itype != torch.int64:
            self.skipTest("COO only supports int64 index type")

        net = SumNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            batch_dim = (
                sparse_input.ndim - sparse_input.sparse_dim() - sparse_input.dense_dim()
            )
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                blocksize = sparse_input.values().shape[batch_dim + 1 : batch_dim + 3]
            else:
                blocksize = None
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertIsInstance(meta, torch.Tensor)
                    self.assertEqual(meta.layout, layout)
                    self.assertEqual(meta.dtype, dtype)
                    (b, s, d, bsz, itp) = extract_sparse_tensor_metadata(meta)
                    self.assertEqual(b, batch_dim)
                    self.assertEqual(s, sparse_input.sparse_dim())
                    self.assertEqual(d, sparse_input.dense_dim())
                    self.assertEqual(bsz, blocksize)
                    self.assertEqual(itp, itype)
                elif i == 1:
                    self.assertIsInstance(meta, FakeTensor)
                    self.assertEqual(meta.layout, torch.strided)
                    self.assertEqual(meta.dtype, dtype)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", SPARSE_DTYPES)
    @parametrize("itype", SPARSE_ITYPES)
    @parametrize("layout", SPARSE_LAYOUTS)
    def test_eltwisenet(self, dtype, itype, layout):
        # TODO: support more cases
        if layout != torch.sparse_coo:
            self.skipTest("layout support not yet implemented")
        if layout == torch.sparse_coo and itype != torch.int64:
            self.skipTest("COO only supports int64 index type")

        net = EltwiseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            batch_dim = (
                sparse_input.ndim - sparse_input.sparse_dim() - sparse_input.dense_dim()
            )
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                blocksize = sparse_input.values().shape[batch_dim + 1 : batch_dim + 3]
            else:
                blocksize = None
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,))
            # Test arg/neg/sin/mul/relu/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i <= 4:
                    self.assertIsInstance(meta, torch.Tensor if i == 0 else FakeTensor)
                    self.assertEqual(meta.layout, layout)
                    self.assertEqual(meta.dtype, dtype)
                    (b, s, d, bsz, itp) = extract_sparse_tensor_metadata(meta)
                    self.assertEqual(b, batch_dim)
                    self.assertEqual(s, sparse_input.sparse_dim())
                    self.assertEqual(d, sparse_input.dense_dim())
                    self.assertEqual(bsz, blocksize)
                    self.assertEqual(itp, itype)
                else:
                    self.assertEqual(meta, None)


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()
