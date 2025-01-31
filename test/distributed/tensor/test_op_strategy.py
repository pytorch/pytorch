# Owner(s): ["oncall: distributed"]

from itertools import chain

import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Replicate,
    Shard,
    TensorMeta,
)
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, PlacementStrategy
from torch.distributed.tensor._ops._einsum_strategy import (
    EinsumDims,
    gen_einsum_strategies,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorOpTestBase


class TestEinsumDims(TestCase):
    def test_batch_dims(self):
        equation = "abc,abc->abc"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["a", "b", "c"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, [])
        self.assertEqual(edims.rhs_out_only_dims, [])

    def test_mm_dims(self):
        equation = "mk,kn->mn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, [])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

    def test_bmm_dims(self):
        equation = "bmk,bkn->bmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b"])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

        equation = "bcmk,bckn->bcmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b", "c"])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

    def test_free_dims(self):
        equation = "abc,ab->abc"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["a", "b"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, ["c"])
        self.assertEqual(edims.rhs_out_only_dims, [])

        equation = "abd,bf->abfd"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, ["a", "d"])
        self.assertEqual(edims.rhs_out_only_dims, ["f"])


class TestEinsumStrategies(DTensorOpTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def test_mm_1d_mesh(self):
        mesh = self.build_device_mesh()

        all_strats = gen_einsum_strategies("mk,kn->mn", mesh)
        self.assertEqual(len(all_strats.strategies), 4)

    def test_mm_2d_mesh(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        all_strats = gen_einsum_strategies("mk,kn->mn", mesh)
        self.assertEqual(len(all_strats.strategies), 16)

    def test_bmm_1d_mesh(self):
        mesh = self.build_device_mesh()

        all_strats = gen_einsum_strategies("bmk,bkn->bmn", mesh)
        self.assertEqual(len(all_strats.strategies), 5)

    def test_bmm_2d_mesh(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        all_strats = gen_einsum_strategies("bmk,bkn->bmn", mesh)
        self.assertEqual(len(all_strats.strategies), 25)

    def test_pointwise_1d_mesh(self):
        mesh = self.build_device_mesh()

        simple_strats = gen_einsum_strategies("abcd,abcd->abcd", mesh)
        self.assertEqual(len(simple_strats.strategies), 5)

        broadcast_strats = gen_einsum_strategies("bcd,abcd->abcd", mesh)
        self.assertEqual(len(broadcast_strats.strategies), 5)

    def test_linearity_1d_mesh(self):
        mesh = self.build_device_mesh()

        all_strats = gen_einsum_strategies("abcd,abcd->abcd", mesh, linearity=True)
        self.assertEqual(len(all_strats.strategies), 6)


class TestCostModel(DTensorOpTestBase):
    def _extract_tensor_meta(self, t) -> TensorMeta:
        return TensorMeta(t.shape, t.stride(), t.dtype)

    @property
    def world_size(self) -> int:
        return 4

    def test_redistribute_cost_mesh_1d(self):
        mesh_1d = self.build_device_mesh()
        shard_placement = (Shard(0),)
        replica_placement = (Replicate(),)
        partial_placement = (Partial(),)

        global_tensor = torch.randn(10, 10)
        global_tensor_meta = self._extract_tensor_meta(global_tensor)

        # shard spec
        shard_spec = DTensorSpec(mesh_1d, shard_placement, global_tensor_meta)
        # replica spec
        replica_spec = DTensorSpec(mesh_1d, replica_placement, global_tensor_meta)
        # partial spec
        partial_spec = DTensorSpec(mesh_1d, partial_placement, global_tensor_meta)

        # make sure reshard cost is 0 for the same spec redistribute
        for spec in [shard_spec, replica_spec, partial_spec]:
            cost = redistribute_cost(spec, spec)
            self.assertEqual(cost, 0)

        # shard -> replicate
        allgather_cost = redistribute_cost(shard_spec, replica_spec)
        # partial -> shard
        reduce_scatter_cost = redistribute_cost(partial_spec, shard_spec)
        # partial -> replicate
        allreduce_cost = redistribute_cost(partial_spec, replica_spec)
        self.assertEqual(allgather_cost, reduce_scatter_cost)
        self.assertTrue(allreduce_cost + 1 < allgather_cost + reduce_scatter_cost)
        # shard to partial
        cost = redistribute_cost(shard_spec, partial_spec)
        self.assertEqual(cost, float("inf"))

    def test_redistribute_cost_latency(self):
        # test cost model on addmm op
        from torch.distributed.tensor._ops._matrix_ops import addmm_strategy

        mesh = self.build_device_mesh()
        shard0_placement = (Shard(0),)
        partial_placement = (Partial(),)
        shard1_placement = (Shard(1),)

        shard0_tensor_meta = self._extract_tensor_meta(torch.randn(8))
        partial_tensor_meta = self._extract_tensor_meta(torch.randn(50, 6))
        shard1_tensor_meta = self._extract_tensor_meta(torch.randn(6, 8))

        # shard spec
        shard0_spec = DTensorSpec(mesh, shard0_placement, shard0_tensor_meta)
        # replica spec
        partial_spec = DTensorSpec(mesh, partial_placement, partial_tensor_meta)
        # partial spec
        shard1_spec = DTensorSpec(mesh, shard1_placement, shard1_tensor_meta)

        op_schema = OpSchema(
            torch.ops.aten.addmm.default,
            (
                OpStrategy([PlacementStrategy(shard0_spec)]),
                OpStrategy([PlacementStrategy(partial_spec)]),
                OpStrategy([PlacementStrategy(shard1_spec)]),
            ),
            {},
        )

        output_strategy = addmm_strategy(mesh, op_schema)
        strategy_costs = {}
        for strategy in output_strategy.strategies:
            redistribute_cost = sum(chain.from_iterable(strategy.redistribute_cost))
            strategy_costs[str(strategy)] = redistribute_cost

        # assert that cost model counts for collective latency (i.e. multiple comm is penalized)
        self.assertTrue(
            strategy_costs["(S(0), R, S(1)) -> S(1)"]
            < strategy_costs["(R, S(0), R) -> S(0)"]
        )
        # assert a single allreduce is the best one
        self.assertEqual(
            strategy_costs["(S(0), R, S(1)) -> S(1)"], min(strategy_costs.values())
        )

    def test_redistribute_cost_mesh_2d(self):
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        shard_placement = (Shard(0), Shard(0))
        replica_placement = (Replicate(), Replicate())
        partial_placement = (Partial(), Partial())

        global_tensor = torch.randn(8, 8)
        global_tensor_meta = self._extract_tensor_meta(global_tensor)

        # shard spec
        shard_spec = DTensorSpec(mesh_2d, shard_placement, global_tensor_meta)
        # replica spec
        replica_spec = DTensorSpec(mesh_2d, replica_placement, global_tensor_meta)
        # partial spec
        partial_spec = DTensorSpec(mesh_2d, partial_placement, global_tensor_meta)

        # make sure reshard cost is 0 for the same spec redistribute
        for spec in [shard_spec, replica_spec, partial_spec]:
            cost = redistribute_cost(spec, spec)
            self.assertEqual(cost, 0)

        # shard -> replicate
        allgather_cost = redistribute_cost(shard_spec, replica_spec)
        # partial -> replicate
        allreduce_cost = redistribute_cost(partial_spec, replica_spec)
        # partial -> shard
        reduce_scatter_cost = redistribute_cost(partial_spec, shard_spec)
        self.assertTrue(allreduce_cost > allgather_cost)
        self.assertTrue(allreduce_cost > reduce_scatter_cost)

    def test_mm_strategies(self):
        from torch.distributed.tensor._ops._matrix_ops import mm_strategy

        mesh = self.build_device_mesh()
        lhs_tensor = torch.randn(6, 8)
        rhs_tensor = torch.randn(8, 12)
        lhs_tensor_meta = self._extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = self._extract_tensor_meta(rhs_tensor)

        mm_combs = (
            (Shard(0), Replicate()),
            (Replicate(), Shard(1)),
            (Shard(1), Shard(0)),
            (Replicate(), Replicate()),
        )
        for lhs, rhs in mm_combs:
            lhs_spec = DTensorSpec(mesh, (lhs,), lhs_tensor_meta)
            rhs_spec = DTensorSpec(mesh, (rhs,), rhs_tensor_meta)

            op_schema = OpSchema(
                torch.ops.aten.mm.default,
                (
                    OpStrategy([PlacementStrategy(lhs_spec)]),
                    OpStrategy([PlacementStrategy(rhs_spec)]),
                ),
                {},
            )
            # test the strategy
            res_strategies = mm_strategy(mesh, op_schema)

            for strtgy in res_strategies.strategies:
                if strtgy.input_specs == (lhs_spec, rhs_spec):
                    self.assertEqual(strtgy.redistribute_cost, [[0.0], [0.0]])
                    break

            op_schema = OpSchema(
                torch.ops.aten.mm.default,
                (lhs_spec, rhs_spec),
                {},
            )
            # test sharding prop
            output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding_non_cached(
                op_schema
            )
            self.assertFalse(output_sharding.needs_redistribute)

    def test_bmm_strategies(self):
        from torch.distributed.tensor._ops._matrix_ops import bmm_strategy

        mesh = self.build_device_mesh()
        lhs_tensor = torch.randn(8, 6, 8)
        rhs_tensor = torch.randn(8, 8, 12)
        lhs_tensor_meta = self._extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = self._extract_tensor_meta(rhs_tensor)

        bmm_combs = (
            (Shard(0), Shard(0)),
            (Shard(1), Replicate()),
            (Replicate(), Shard(2)),
            (Shard(2), Shard(1)),
            (Replicate(), Replicate()),
        )
        for lhs, rhs in bmm_combs:
            lhs_spec = DTensorSpec(mesh, (lhs,), lhs_tensor_meta)
            rhs_spec = DTensorSpec(mesh, (rhs,), rhs_tensor_meta)

            op_schema = OpSchema(
                torch.ops.aten.bmm.default,
                (
                    OpStrategy([PlacementStrategy(lhs_spec)]),
                    OpStrategy([PlacementStrategy(rhs_spec)]),
                ),
                {},
            )
            # test the strategy
            res_strategies = bmm_strategy(mesh, op_schema)

            for strtgy in res_strategies.strategies:
                if strtgy.input_specs == (lhs_spec, rhs_spec):
                    self.assertEqual(strtgy.redistribute_cost, [[0.0], [0.0]])
                    break

            op_schema = OpSchema(
                torch.ops.aten.bmm.default,
                (lhs_spec, rhs_spec),
                {},
            )
            # test sharding prop
            output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding_non_cached(
                op_schema
            )
            self.assertFalse(output_sharding.needs_redistribute)


if __name__ == "__main__":
    run_tests()
