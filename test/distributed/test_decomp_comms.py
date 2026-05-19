# Owner(s): ["module: inductor"]
"""
Unit tests for decomp_gram_matrix_all_gather FX pass.

Verifies the pass transforms:
    all_gather(X_shard) -> wait -> slice -> [Gram compute] -> split -> getitem
into:
    X_shard -> [local compute + all_reduce for Gram aggregation]
"""

import operator
import unittest

import torch
import torch.distributed as dist
import torch.fx as fx

from torch._inductor.fx_passes.decomp_comms import decomp_gram_matrix_all_gather
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_GPU

aten = torch.ops.aten
c10d = torch.ops._c10d_functional


def _count_ops(graph: fx.Graph, target) -> int:
    return sum(1 for n in graph.nodes if n.op == "call_function" and n.target is target)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestDecompGramMatrixAllGather(InductorTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        dist.destroy_process_group()

    def test_muon_newton_schulz(self):
        """
        Muon optimizer: Newton-Schulz with X @ X.T Gram, 3 iterations.
        Verifies all_gather removed, one all_reduce inserted per iteration.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def muon_step(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            x = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0])

            norm_val = aten.clamp_min.default(
                aten.pow.Tensor_Scalar(
                    aten.sum.dim_IntList(aten.pow.Tensor_Scalar(x, 2), [0, 1]),
                    0.5,
                ),
                1e-7,
            )
            x = aten.div.Tensor(x, norm_val)

            for _ in range(ns_steps):
                xt = aten.permute.default(x, [1, 0])
                gram = aten.mm.default(x, xt)  # X @ X.T — Gram
                gram_sq = aten.mm.default(gram, gram)
                update = aten.add.Tensor(
                    aten.mul.Tensor(gram, -4.7750),
                    aten.mul.Tensor(gram_sq, 2.0315),
                )
                x = aten.add.Tensor(
                    aten.mul.Tensor(x, 3.4445),
                    aten.mm.default(update, x),
                )

            chunks = aten.split.Tensor(x, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(32, 128, device=self.device)
            traced = make_fx(muon_step)(x_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 0
        )
        self.assertEqual(
            _count_ops(traced.graph, c10d.all_reduce.default), ns_steps
        )
        self.assertEqual(_count_ops(traced.graph, aten.split.Tensor), 0)

    def test_shampoo_preconditioner(self):
        """
        Shampoo optimizer: G.T @ G right Kronecker factor.
        Verifies the pass handles both Gram directions (X.T @ X, not just X @ X.T).
        """
        group_name = "0"
        world_size = 2

        def shampoo_step(g_shard):
            gathered = c10d.all_gather_into_tensor.default(
                g_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            g = aten.slice.Tensor(waited, 0, 0, g_shard.shape[0])

            gt = aten.permute.default(g, [1, 0])
            gram = aten.mm.default(gt, g)  # G.T @ G — Gram

            precond = aten.mul.Tensor(gram, -0.5)
            g_precond = aten.mm.default(g, precond)

            chunks = aten.split.Tensor(g_precond, g_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            g_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(shampoo_step)(g_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 0
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 1)
        self.assertEqual(_count_ops(traced.graph, aten.split.Tensor), 0)

    def test_no_transform_without_gram(self):
        """
        Pass must NOT transform when no Gram mm is detected.
        mm(X, W) with independent W is not a Gram pattern.
        """
        group_name = "0"
        world_size = 2

        def no_gram(x_shard, weight):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            x = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0])

            out = aten.mm.default(x, weight)

            chunks = aten.split.Tensor(out, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(32, 64, device=self.device)
            weight = torch.randn(64, 16, device=self.device)
            traced = make_fx(no_gram)(x_shard, weight)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1,
            "all_gather should remain when no Gram pattern detected",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)


if __name__ == "__main__":
    run_tests()
