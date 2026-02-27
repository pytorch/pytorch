# Owner(s): ["module: sparse"]
#
# Test to ensure sparsity information propagates properly into traced graph.
#

import unittest

import torch
from torch._environment import is_fbcode
from torch._subclasses.fake_tensor import FakeTensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


# Various data types (preserved over operations).
DTYPES = [
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]

# Various index types.
ITYPES = [torch.int32, torch.int64]


# Constructs a subtest for every sparse layout currently supported in torch.sparse.
def all_sparse_layouts(test_name="layout"):
    return parametrize(
        test_name,
        [
            subtest(torch.sparse_coo, name="SparseCOO"),
            subtest(torch.sparse_csr, name="SparseCSR"),
            subtest(torch.sparse_csc, name="SparseCSC"),
            subtest(torch.sparse_bsr, name="SparseBSR"),
            subtest(torch.sparse_bsc, name="SparseBSC"),
        ],
    )


#
# Various network examples.
#


class IdNet(torch.nn.Module):
    def forward(self, x):
        return x


class SumNet(torch.nn.Module):
    def forward(self, x):
        return x.sum()


class EltwiseNet(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu(2 * torch.abs(-x))


class ToDenseNet(torch.nn.Module):
    def forward(self, x):
        return x.to_dense()


class AddNet(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)


class SparseActivationCOO(torch.nn.Module):
    def forward(self, x):
        return [xi.to_sparse() for xi in x]


class SparseActivationCSR(torch.nn.Module):
    def forward(self, x):
        return [xi.to_sparse_csr() for xi in x]


#
# The test driver.
#


@unittest.skipIf(is_fbcode(), "See torch._dynamo.config")
class TestSparseProp(TestCase):
    def setUp(self):
        super().setUp()

    def assertEqualMeta(self, x, y):
        self.assertIsInstance(x, FakeTensor)
        self.assertIsInstance(y, torch.Tensor)

        # Convert expected value to meta for comparison.
        y = y.to("meta")
        self.assertEqual(x, y, exact_layout=True, exact_is_coalesced=True)

        # When x or y is a meta tensor (say, `x.device == "meta"`), then
        # assertEqual(x, y) compares only x and y attributes but skips
        # comparing their values. In the case of sparse tensors, this means
        # that comparing indices and values attributes are skipped as well,
        # which is why we are doing that explicitly below.
        if x.layout is torch.strided:
            pass
        elif x.layout is torch.sparse_coo:
            self.assertEqual(x._indices(), y._indices(), exact_layout=True)
            self.assertEqual(x._values(), y._values(), exact_layout=True)
        else:
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
                x_meta1, y_meta1 = (x.crow_indices(), y.crow_indices())
                x_meta2, y_meta2 = (x.col_indices(), y.col_indices())
            elif x.layout in {torch.sparse_csc, torch.sparse_bsc}:
                x_meta1, y_meta1 = (x.ccol_indices(), y.ccol_indices())
                x_meta2, y_meta2 = (x.row_indices(), y.row_indices())
            else:
                raise AssertionError(f"Unexpected layout: {x.layout}")
            self.assertEqual(x_meta1, y_meta1, exact_layout=True)
            self.assertEqual(x_meta2, y_meta2, exact_layout=True)
            self.assertEqual(x.values(), y.values(), exact_layout=True)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_idnet(self, dtype, itype, layout):
        net = IdNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_sumnet(self, dtype, itype, layout):
        net = SumNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                elif i == 1:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_eltwisenet(self, dtype, itype, layout):
        net = EltwiseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/neg/abs/mul/relu/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i <= 4:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_todensenet(self, dtype, itype, layout):
        net = ToDenseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/todense/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                elif i == 1:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    def test_add(self):
        net = AddNet()
        Y = torch.arange(16, 32, dtype=torch.float32).view(4, 4)
        A = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        S = A.to_sparse_csr()
        result = net(S, Y)
        # Build the traced graph.
        prog = torch.export.export(net, (S, Y), strict=True)
        # Test args/add/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i == 0:
                self.assertEqualMeta(meta, S)
            elif i == 1:
                self.assertEqualMeta(meta, Y)
            elif i == 2:
                self.assertEqualMeta(meta, result)
            else:
                self.assertEqual(meta, None)

    def test_activation_coo(self):
        net = SparseActivationCOO()
        x = [torch.randn(3, 3) for _ in range(3)]
        result = net(x)
        # Build the traced graph.
        prog = torch.export.export(net, args=(x,), strict=True)
        # Test args/to_sparse/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i <= 2:
                self.assertEqualMeta(meta, x[i])
            elif i <= 5:
                self.assertEqualMeta(meta, result[i - 3])
            else:
                self.assertEqual(meta, None)

    def test_activation_csr(self):
        net = SparseActivationCSR()
        x = [torch.randn(3, 3) for _ in range(3)]
        result = net(x)
        # Build the traced graph.
        prog = torch.export.export(net, args=(x,), strict=True)
        # Test args/to_sparse/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i <= 2:
                self.assertEqualMeta(meta, x[i])
            elif i <= 5:
                self.assertEqualMeta(meta, result[i - 3])
            else:
                self.assertEqual(meta, None)


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()
