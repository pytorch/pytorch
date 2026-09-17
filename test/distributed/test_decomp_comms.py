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
from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_GPU


aten = torch.ops.aten
c10d = torch.ops._c10d_functional


def _count_ops(graph: fx.Graph, target) -> int:  # type: ignore[type-arg]
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
        Rigi Muon optimizer pattern with initial transpose (rows > cols).

        Shard (S, K) where S*W > K. After all_gather: (S*W, K).
        Muon transposes to (K, S*W), then Gram = X @ X.T = (K, K).
        The Gram output (K, K) does NOT depend on gathered dim (S*W),
        so the decomposition is shape-correct.

        After transpose back and scale, result is (S*W, K), split to (S, K).
        """
        group_name = "0"
        world_size = 2
        ns_steps = 5

        def muon_step(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            # Identity slice — keeps full gathered tensor
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            X = aten._to_copy.default(X, dtype=torch.bfloat16)
            norm_val = aten.linalg_vector_norm.default(X, 2, [-2, -1], True)
            norm_val = aten.clamp_min.default(norm_val, 1e-7)
            X = aten.div.Tensor(X, norm_val)

            # Transpose: (S*W, K) -> (K, S*W), Gram will be (K, K)
            X = aten.transpose.int(X, -2, -1)

            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                B = aten.add.Tensor(
                    aten.mul.Tensor(A, -4.7750),
                    aten.mul.Tensor(aten.mm.default(A, A), 2.0315),
                )
                X = aten.add.Tensor(
                    aten.mul.Tensor(X, 3.4445),
                    aten.mm.default(B, X),
                )

            # Transpose back: (K, S*W) -> (S*W, K)
            X = aten.transpose.int(X, -2, -1)
            X = aten.mul.Tensor(X, 0.2 * (512**0.5))

            # Split back to shard
            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            # S=64, K=32. Gathered=(128,32). After transpose X=(32,128).
            # Gram=(32,32). gathered_dim=128, gram_dim=32 → decomposable.
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(muon_step)(x_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 0
        )
        # 5 all_reduces for Gram mms + 1 for the norm reduction
        self.assertGreaterEqual(
            _count_ops(traced.graph, c10d.all_reduce.default), ns_steps
        )
        self.assertEqual(_count_ops(traced.graph, aten.split.Tensor), 0)

    def test_no_slice_gathered_consumed_directly(self):
        """
        Resilient anchor: when inductor folds the FSDP no-op identity slice
        (even sharding), the gathered tensor is consumed straight off the
        wait, with no aten.slice. The pass must still anchor on
        wait(all_gather) and transform. The wait here also has multiple direct
        users (norm + div), which the previous wait.users==1 gate rejected.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def muon_no_slice(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            # No slice: wait feeds the chain directly (norm + div).
            norm_val = aten.linalg_vector_norm.default(waited, 2, [-2, -1], True)
            norm_val = aten.clamp_min.default(norm_val, 1e-7)
            X = aten.div.Tensor(waited, norm_val)

            X = aten.transpose.int(X, -2, -1)
            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))
            X = aten.transpose.int(X, -2, -1)

            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(muon_no_slice)(x_shard)

        # Precondition: the gathered tensor is consumed with no slice.
        self.assertEqual(_count_ops(traced.graph, aten.slice.Tensor), 0)
        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            0,
            "all_gather should be eliminated even with no slice wrapping the wait",
        )
        self.assertGreaterEqual(
            _count_ops(traced.graph, c10d.all_reduce.default), ns_steps
        )
        self.assertEqual(_count_ops(traced.graph, aten.split.Tensor), 0)

    def test_no_transform_unsafe_entry_region(self):
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def finish(X, x_shard):
            X = aten.transpose.int(X, -2, -1)
            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))
            X = aten.transpose.int(X, -2, -1)
            return operator.getitem(aten.split.Tensor(X, x_shard.shape[0], 0), 0)

        def side_user(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            side = aten.mul.Tensor(waited, 2.0)
            return finish(waited, x_shard), side

        def column_slice(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            return finish(aten.slice.Tensor(waited, 1, 0, 16), x_shard)

        def unsupported_reduction(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            return finish(
                aten.sub.Tensor(waited, aten.amax.default(waited, [0], True)), x_shard
            )

        for fn in (side_user, column_slice, unsupported_reduction):
            with FakeTensorMode():
                traced = make_fx(fn)(torch.randn(64, 32, device=self.device))

            decomp_gram_matrix_all_gather(traced)

            self.assertEqual(
                _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
            )
            self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_shampoo_preconditioner(self):
        """
        Shampoo-like optimizer with iterative preconditioner (3 iterations).
        G.T @ G right Kronecker factor via aten.permute.
        Multiple Gram mms per all_gather to pass the NS-pattern threshold.
        """
        group_name = "0"
        world_size = 2
        n_iters = 3

        def shampoo_step(g_shard):
            gathered = c10d.all_gather_into_tensor.default(
                g_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            g = aten.slice.Tensor(waited, 0, 0, g_shard.shape[0] * world_size)

            # Iterative preconditioner (coupled Newton for matrix sqrt)
            precond = g
            for _ in range(n_iters):
                gt = aten.permute.default(precond, [1, 0])
                gram = aten.mm.default(gt, precond)  # G.T @ G = (K, K)
                precond = aten.sub.Tensor(precond, aten.mm.default(precond, gram))

            g_precond = aten.mm.default(g, aten.permute.default(precond, [1, 0]))

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
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), n_iters)
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
            x = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            out = aten.mm.default(x, weight)

            chunks = aten.split.Tensor(out, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(32, 64, device=self.device)
            weight = torch.randn(64, 16, device=self.device)
            traced = make_fx(no_gram)(x_shard, weight)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain when no Gram pattern detected",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_no_transform_non_decomposable_shape(self):
        """
        mm(X, X.T) where X is (N, N) square: Gram output (N, N) depends
        on the gathered dim, so the decomposition is NOT shape-correct.
        Pass must not transform.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def square_gram(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            for _ in range(ns_steps):
                # mm(X, X.T): output is (N, N) where N = gathered rows
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))

            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            # Shard (32, 64), gathered (64, 64) -- square.
            # Gram = mm(X, X.T) = (64, 64), depends on gathered dim.
            x_shard = torch.randn(32, 64, device=self.device)
            traced = make_fx(square_gram)(x_shard)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain for non-decomposable Gram shape",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_no_transform_single_gram_mm(self):
        """
        Single Gram mm doesn't meet the _MIN_GRAM_MMS=2 threshold.
        Pass must not transform (likely forward/backward, not iterative optimizer).
        """
        group_name = "0"
        world_size = 2

        def single_gram(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            Xt = aten.transpose.int(X, -2, -1)
            gram = aten.mm.default(Xt, X)
            out = aten.mm.default(X, gram)

            chunks = aten.split.Tensor(out, x_shard.shape[0], 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(single_gram)(x_shard)

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain with only 1 Gram mm (below threshold)",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)

    def test_padded_shard_trimming(self):
        """
        FSDP padding: the shard is padded for divisibility, but the gathered
        tensor is sliced down to the unpadded total before the Gram compute
        and split into unpadded per-rank chunks. The getitem output
        (unpadded_per_rank) is then smaller than the padded shard, so the
        pass must insert a slice trimming the shard down to the unpadded size.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3
        shard_size = 64  # padded shard
        unpadded_per_rank = 60
        gathered_unpadded = unpadded_per_rank * world_size

        def padded_step(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            # Slice strips padding: keep only the unpadded total rows.
            X = aten.slice.Tensor(waited, 0, 0, gathered_unpadded)

            X = aten.transpose.int(X, -2, -1)

            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))

            X = aten.transpose.int(X, -2, -1)

            # Even split into unpadded per-rank chunks (60 < padded 64).
            chunks = aten.split.Tensor(X, unpadded_per_rank, 0)
            return operator.getitem(chunks, 0)

        with FakeTensorMode():
            x_shard = torch.randn(shard_size, 32, device=self.device)
            traced = make_fx(padded_step)(x_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 0
        )
        self.assertGreaterEqual(
            _count_ops(traced.graph, c10d.all_reduce.default), ns_steps
        )
        # Trim path under test: pass slices the placeholder shard to unpadded size.
        trim_slices = [
            n
            for n in traced.graph.nodes
            if n.op == "call_function"
            and n.target is aten.slice.Tensor
            and isinstance(n.args[0], fx.Node)
            and n.args[0].op == "placeholder"
            and len(n.args) >= 4
            and n.args[3] == unpadded_per_rank
        ]
        self.assertEqual(len(trim_slices), 1, "pass should insert a shard-trim slice")

    def test_no_transform_multiple_consumed_getitems(self):
        """
        When the split feeds multiple consumed getitems (e.g. several ranks'
        shards used in the same graph), collapsing the split into a single
        rank-local result would leave the sibling getitems reading past the
        now shard-sized split input. The pass must bail rather than corrupt
        the graph.
        """
        group_name = "0"
        world_size = 2
        ns_steps = 3

        def multi_user_step(x_shard):
            gathered = c10d.all_gather_into_tensor.default(
                x_shard, world_size, group_name
            )
            waited = c10d.wait_tensor.default(gathered)
            X = aten.slice.Tensor(waited, 0, 0, x_shard.shape[0] * world_size)

            X = aten.transpose.int(X, -2, -1)

            for _ in range(ns_steps):
                A = aten.mm.default(X, aten.transpose.int(X, -2, -1))
                X = aten.add.Tensor(X, aten.mm.default(A, X))

            X = aten.transpose.int(X, -2, -1)

            chunks = aten.split.Tensor(X, x_shard.shape[0], 0)
            # Both getitems have users
            c0 = operator.getitem(chunks, 0)
            c1 = operator.getitem(chunks, 1)
            return aten.add.Tensor(c0, c1)

        with FakeTensorMode():
            x_shard = torch.randn(64, 32, device=self.device)
            traced = make_fx(multi_user_step)(x_shard)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default), 1
        )

        decomp_gram_matrix_all_gather(traced)

        self.assertEqual(
            _count_ops(traced.graph, c10d.all_gather_into_tensor.default),
            1,
            "all_gather should remain when split has other consumed getitems",
        )
        self.assertEqual(_count_ops(traced.graph, c10d.all_reduce.default), 0)


if __name__ == "__main__":
    run_tests()
