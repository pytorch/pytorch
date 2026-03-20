# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch._decomp import register_decomposition
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestDecompSharding(TestCase):
    world_size = 4

    def setUp(self):
        super().setUp()
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_aminmax_decomp(self):
        """
        def decomp(x):
            amin = aten.amin.default(x)
            amax = aten.amax.default(x)
            return return_types_aminmax(amin, amax)
        """
        from torch.distributed.tensor import empty as d_empty

        # 1d mesh
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        x_local = torch.randn(16)
        x = DTensor.from_local(x_local, mesh, [Shard(0)], run_check=False)
        out = torch.aminmax(x)
        self.assertEqual(out.min.placements, (Partial("min"),))
        self.assertEqual(out.max.placements, (Partial("max"),))

        # 2d mesh
        mesh = DeviceMesh("cpu", torch.arange(self.world_size).reshape(-1, 2))

        x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(1), Partial("min")])
        out = torch.aminmax(x)
        self.assertEqual(out.min.placements, (Partial("min"), Partial("min")))
        self.assertEqual(out.max.placements, (Partial("max"), Partial("max")))

    def test_custom_recursive_decomp(self):
        """
        op1 decomps -> op2, which decomps -> mm

        def op1(x, y):
            return op2(x, y) * 1.0

        def op2(x, y):
            return x @ y

        We also test that sharding prop caching kicks in for decompositions;
        1) calling op1 twice should cache hit
        2) since op1 runs op2's decomposition, calling op2 after should also cache hit
        """
        from torch.distributed.tensor import empty as d_empty
        from torch.distributed.tensor.debug import _get_python_sharding_prop_cache_info

        with torch.library._scoped_library("sharding_decomps", "FRAGMENT") as my_lib:
            my_lib.define("op1(Tensor x, Tensor y) -> Tensor")
            my_lib.define("op2(Tensor x, Tensor y) -> Tensor")

            @torch.library.impl(my_lib, "op1", "CPU")
            def op1_impl(x, y):
                return x @ y * 1.0

            @torch.library.impl(my_lib, "op2", "CPU")
            def op2_impl(x, y):
                return x @ y

            @register_decomposition(torch.ops.sharding_decomps.op1.default)
            def op1_decomp(x, y):
                return torch.ops.sharding_decomps.op2(x, y) * 1.0

            @register_decomposition(torch.ops.sharding_decomps.op2.default)
            def op2_decomp(x, y):
                return x @ y

            @torch.library.register_fake("sharding_decomps::op1")
            def op1_meta(x, y):
                return torch.empty_like(x) @ torch.empty_like(y)

            @torch.library.register_fake("sharding_decomps::op2")
            def op2_meta(x, y):
                return torch.empty_like(x) @ torch.empty_like(y)

            mesh = DeviceMesh("cpu", torch.arange(self.world_size).reshape(-1, 2))
            x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(0), Shard(1)])
            y = d_empty(16, 16, device_mesh=mesh, placements=[Replicate(), Shard(0)])

            # op1 1st call
            out = torch.ops.sharding_decomps.op1(x, y)

            # expect matmul placements
            self.assertEqual(out.placements, (Shard(0), Partial("sum")))

            # starting cache size
            cache = _get_python_sharding_prop_cache_info()
            cache_size = cache.currsize

            # op1 2nd call
            torch.ops.sharding_decomps.op1(x, y)
            cache = _get_python_sharding_prop_cache_info()
            self.assertEqual(cache_size, cache.currsize)

            # op2 1st call, expect same placements + cache hit
            out = torch.ops.sharding_decomps.op2(x, y)
            self.assertEqual(out.placements, (Shard(0), Partial("sum")))
            cache = _get_python_sharding_prop_cache_info()
            self.assertEqual(cache_size, cache.currsize)

    def test_misc_ops_with_no_sharding_rules(self):
        """miscellaneous aten ops"""
        from torch.distributed.tensor import empty as d_empty

        aten = torch.ops.aten
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))
        mesh_2d = DeviceMesh("cpu", torch.arange(self.world_size).reshape(-1, 2))

        def check_no_strategy(op):
            # if someone registers a rule for these ops, delete the test
            sharding_prop = DTensor._op_dispatcher.sharding_propagator
            self.assertTrue(op not in sharding_prop.op_strategy_funcs)
            self.assertTrue(op not in sharding_prop.op_single_dim_strategy_funcs)
            self.assertTrue(op not in sharding_prop.op_to_rules)

        # binary_cross_entropy_with_logits
        check_no_strategy(aten.binary_cross_entropy_with_logits.default)
        input = d_empty(16, device_mesh=mesh, placements=[Shard(0)])
        target = d_empty(16, device_mesh=mesh, placements=[Shard(0)])
        weight = d_empty(16, device_mesh=mesh, placements=[Shard(0)])
        out = aten.binary_cross_entropy_with_logits.default(input, target, weight)
        self.assertEqual(out.placements, (Partial("avg"),))

        # mse_loss
        check_no_strategy(aten.mse_loss.default)
        input = d_empty(16, device_mesh=mesh, placements=[Shard(0)])
        target = d_empty(16, device_mesh=mesh, placements=[Shard(0)])
        out = aten.mse_loss.default(input, target)
        self.assertEqual(out.placements, (Partial("avg"),))

        # smooth_l1_loss
        check_no_strategy(aten.smooth_l1_loss.default)
        input = d_empty(
            16,
            device_mesh=mesh_2d,
            placements=[_StridedShard(0, split_factor=2), Shard(0)],
        )
        target = d_empty(
            16,
            device_mesh=mesh_2d,
            placements=[_StridedShard(0, split_factor=2), Shard(0)],
        )
        out = aten.smooth_l1_loss.default(input, target)
        self.assertEqual(out.placements, (Partial("avg"), Partial("avg")))

        # expand_copy: has a registered strategy (same as expand)
        input = d_empty(16, 1, device_mesh=mesh, placements=[Partial("min")])
        out = aten.expand_copy.default(input, [-1, 16])
        self.assertEqual(out.placements, (Partial("min"),))

        # glu: force replicate
        check_no_strategy(aten.glu.default)
        x = d_empty(16, device_mesh=mesh, placements=[Partial()])
        out = aten.glu.default(x)
        self.assertEqual(out.placements, (Replicate(),))

        # index_add: decomposes into index_put with accumulate=True
        check_no_strategy(aten.index_add.default)
        input = d_empty(4, 8, device_mesh=mesh, placements=[Shard(1)])
        index = distribute_tensor(torch.tensor([0, 2]), mesh, [Replicate()])
        source = d_empty(2, 8, device_mesh=mesh, placements=[Shard(1)])
        out = aten.index_add.default(input, 0, index, source)
        self.assertEqual(out.placements, (Shard(1),))

        # polar: force replicate
        check_no_strategy(aten.polar.default)
        x = d_empty(16, device_mesh=mesh, placements=[Partial()])
        y = d_empty(16, device_mesh=mesh, placements=[Partial()])
        out = aten.polar.default(x, y)
        self.assertEqual(out.placements, (Replicate(),))

    def test_roll_flip_strategies(self):
        """roll and flip unshard on active dims, keep sharding on others."""
        from torch.distributed.tensor import empty as d_empty

        aten = torch.ops.aten
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        # roll: sharded on non-roll dim stays sharded
        x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(0)])
        out = aten.roll.default(x, [2], [1])
        self.assertEqual(out.placements, (Shard(0),))

        # roll with no dims (flattened): always replicates
        x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(0)])
        out = aten.roll.default(x, [2], [])
        self.assertEqual(out.placements, (Replicate(),))

        # flip: sharded on non-flip dim stays sharded
        x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(0)])
        out = aten.flip.default(x, [1])
        self.assertEqual(out.placements, (Shard(0),))

        # flip: sharded on flip dim gets replicated
        x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(0)])
        out = aten.flip.default(x, [0])
        self.assertEqual(out.placements, (Replicate(),))

    def test_fft_strategies(self):
        """FFT primitives unshard on transform dims, keep sharding on others."""
        from torch.distributed.tensor import empty as d_empty

        aten = torch.ops.aten
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        # _fft_c2c: sharded on non-transform dim stays sharded
        x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(0)], dtype=torch.cfloat)
        out = aten._fft_c2c.default(x, [1], 0, True)
        self.assertEqual(out.placements, (Shard(0),))

        # _fft_c2c: sharded on transform dim gets replicated
        x = d_empty(16, 16, device_mesh=mesh, placements=[Shard(1)], dtype=torch.cfloat)
        out = aten._fft_c2c.default(x, [1], 0, True)
        self.assertEqual(out.placements, (Replicate(),))


class TestDecompShardingWithComms(DTensorTestBase):
    @with_comms
    def test_decomp_schema_caches_static_args(self):
        """
        Test that decomposition ops use the correct cache key with static args.
        unsafe_chunk decomposes through split, and two unsafe_chunk calls with different dims
        could cache hit without correct dim/start/end handling.

        This checks the first call allows Shard(2) through, while the 2nd call forces Replicate.
        """
        device_mesh = self.build_device_mesh()
        t = torch.randn(8, 8, 8, requires_grad=False, device=self.device_type)
        dt = distribute_tensor(t, device_mesh, [Shard(2)])

        # chunk on non-sharding dim propagates through
        result_dim1 = torch.unsafe_chunk(dt, 2, dim=1)
        expected_dim1 = torch.unsafe_chunk(t, 2, dim=1)
        for r, e in zip(result_dim1, expected_dim1):
            self.assertEqual(r.placements, (Shard(2),))
            self.assertEqual(r.full_tensor(), e)

        # chunk on sharding dim forces replicate
        result_dim2 = torch.unsafe_chunk(dt, 2, dim=2)
        expected_dim2 = torch.unsafe_chunk(t, 2, dim=2)
        for r, e in zip(result_dim2, expected_dim2):
            self.assertEqual(r.placements, (Replicate(),))
            self.assertEqual(r.full_tensor(), e)

    @with_comms
    def test_decomp_schema_for_cache_aminmax(self):
        """Test that aminmax with different dim kwargs doesn't hit stale cache."""
        device_mesh = self.build_device_mesh()
        t = torch.randn(
            self.world_size, 4, 4, requires_grad=False, device=self.device_type
        )
        dt = distribute_tensor(t, device_mesh, [Shard(0)])

        # First call: full reduction (no dim kwarg)
        result_full = torch.aminmax(dt)
        expected_full = torch.aminmax(t)
        self.assertEqual(result_full.min.shape, expected_full.min.shape)
        self.assertEqual(result_full.min.full_tensor(), expected_full.min)

        # Second call: partial reduction with dim=1 kwarg
        # This should NOT reuse the cached result from the first call
        result_dim1 = torch.aminmax(dt, dim=1)
        expected_dim1 = torch.aminmax(t, dim=1)
        self.assertTrue(torch.equal(result_dim1.min.full_tensor(), expected_dim1.min))
        self.assertTrue(torch.equal(result_dim1.max.full_tensor(), expected_dim1.max))

    def _remove_strategy(self, op):
        """Temporarily remove an op's explicit strategy to test DecompSharding fallback."""
        prop = DTensor._op_dispatcher.sharding_propagator
        saved = {}
        for name, d in [
            ("op_strategy", prop.op_strategy_funcs),
            ("op_schema", prop.op_to_schema_info),
            ("single_dim", prop.op_single_dim_strategy_funcs),
            ("single_dim_schema", prop.op_to_schema_info_for_single_dim_strategy),
        ]:
            if op in d:
                saved[name] = d.pop(op)
        if hasattr(prop.propagate_op_sharding, "cache_clear"):
            prop.propagate_op_sharding.cache_clear()
        return saved

    def _restore_strategy(self, op, saved):
        prop = DTensor._op_dispatcher.sharding_propagator
        mapping = {
            "op_strategy": prop.op_strategy_funcs,
            "op_schema": prop.op_to_schema_info,
            "single_dim": prop.op_single_dim_strategy_funcs,
            "single_dim_schema": prop.op_to_schema_info_for_single_dim_strategy,
        }
        for name, val in saved.items():
            mapping[name][op] = val
        if hasattr(prop.propagate_op_sharding, "cache_clear"):
            prop.propagate_op_sharding.cache_clear()

    def _test_decomp_fallback(self, op, run_fn):
        """Remove explicit strategy, run op via DecompSharding, verify it returns DTensor."""
        saved = self._remove_strategy(op)
        try:
            result = run_fn()
            if isinstance(result, (tuple, list)):
                self.assertTrue(any(isinstance(r, DTensor) for r in result))
            else:
                self.assertIsInstance(result, DTensor)
        finally:
            self._restore_strategy(op, saved)

    @with_comms
    def test_decomp_fallback_ops(self):
        """Ops with decompositions that work via DecompSharding without explicit strategies."""
        aten = torch.ops.aten
        mesh = self.build_device_mesh()
        dt = lambda t: DTensor.from_local(t, mesh, [Replicate()])

        x = dt(torch.randn(4, 5, device=self.device_type))
        idx = dt(torch.tensor([0, 2], device=self.device_type))
        src = dt(torch.randn(2, 5, device=self.device_type))

        self._test_decomp_fallback(
            aten.index_add.default, lambda: aten.index_add(x, 0, idx, src)
        )
        self._test_decomp_fallback(
            aten.index_copy.default, lambda: aten.index_copy(x, 0, idx, src)
        )
        val = dt(torch.tensor(3.14, device=self.device_type))
        self._test_decomp_fallback(
            aten.index_fill.int_Tensor,
            lambda: aten.index_fill.int_Tensor(x, 0, idx, val),
        )
        self._test_decomp_fallback(
            aten.isin.Tensor_Scalar,
            lambda: aten.isin.Tensor_Scalar(
                dt(torch.tensor([1, 2, 3], device=self.device_type)), 3
            ),
        )
        self._test_decomp_fallback(
            aten.renorm.default, lambda: aten.renorm(x, 2.0, 0, 1.0)
        )
        self._test_decomp_fallback(
            aten.kron.default,
            lambda: aten.kron(
                dt(torch.randn(2, 3, device=self.device_type)),
                dt(torch.randn(4, 5, device=self.device_type)),
            ),
        )

    @with_comms
    def test_new_decomp_addbmm(self):
        aten = torch.ops.aten
        mesh = self.build_device_mesh()
        dt = lambda t: DTensor.from_local(t, mesh, [Replicate()])
        a = torch.randn(4, 6, device=self.device_type)
        b1 = torch.randn(2, 4, 3, device=self.device_type)
        b2 = torch.randn(2, 3, 6, device=self.device_type)
        ref = torch.addbmm(a, b1, b2)
        out = aten.addbmm(dt(a), dt(b1), dt(b2))
        self.assertIsInstance(out, DTensor)
        self.assertEqual(ref, out.full_tensor())

    @with_comms
    def test_new_decomp_conj_physical(self):
        aten = torch.ops.aten
        mesh = self.build_device_mesh()
        dt = lambda t: DTensor.from_local(t, mesh, [Replicate()])
        x = torch.randn(3, 4, dtype=torch.cfloat, device=self.device_type)
        out = aten._conj_physical(dt(x))
        self.assertIsInstance(out, DTensor)

    @with_comms
    def test_new_decomp_rrelu(self):
        aten = torch.ops.aten
        mesh = self.build_device_mesh()
        dt = lambda t: DTensor.from_local(t, mesh, [Replicate()])
        x = dt(torch.randn(3, 4, device=self.device_type))
        noise = dt(torch.empty(3, 4, device=self.device_type))
        out = aten.rrelu_with_noise(x, noise, 0.125, 0.333, False)
        self.assertIsInstance(out, DTensor)

    @with_comms
    def test_new_decomp_pdist(self):
        aten = torch.ops.aten
        mesh = self.build_device_mesh()
        dt = lambda t: DTensor.from_local(t, mesh, [Replicate()])
        x = dt(torch.randn(8, 5, device=self.device_type))
        out = aten._pdist_forward(x, 2.0)
        self.assertIsInstance(out, DTensor)
        self.assertEqual(out.shape, torch.Size([28]))

    @with_comms
    def test_new_decomp_cdist(self):
        aten = torch.ops.aten
        mesh = self.build_device_mesh()
        dt = lambda t: DTensor.from_local(t, mesh, [Replicate()])
        x1 = torch.randn(2, 5, 3, device=self.device_type)
        x2 = torch.randn(2, 4, 3, device=self.device_type)
        ref = torch.cdist(x1, x2, 2.0)
        out = torch.cdist(dt(x1), dt(x2), 2.0)
        self.assertIsInstance(out, DTensor)
        self.assertEqual(ref, out.full_tensor(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
