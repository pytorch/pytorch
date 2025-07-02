# Owner(s): ["oncall: distributed"]

import functools
import re
import warnings
from itertools import chain
from unittest.mock import patch

import numpy as np

import torch
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    StrategyType,
)
from torch.distributed.tensor._ops._einsum_strategy import (
    EinsumDims,
    gen_einsum_strategies,
)
from torch.distributed.tensor._ops.utils import register_op_strategy
from torch.distributed.tensor._utils import generate_redistribute_costs
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorOpTestBase,
    DTensorTestBase,
    with_comms,
)


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
                OpStrategy([OpSpec(shard0_spec)]),
                OpStrategy([OpSpec(partial_spec)]),
                OpStrategy([OpSpec(shard1_spec)]),
            ),
            {},
        )

        output_strategy = addmm_strategy(op_schema)
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
                    OpStrategy([OpSpec(lhs_spec)]),
                    OpStrategy([OpSpec(rhs_spec)]),
                ),
                {},
            )
            # test the strategy
            res_strategies = mm_strategy(op_schema)

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
                    OpStrategy([OpSpec(lhs_spec)]),
                    OpStrategy([OpSpec(rhs_spec)]),
                ),
                {},
            )
            # test the strategy
            res_strategies = bmm_strategy(op_schema)

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


# Test op strategy registration


# Dummy ops to test op strategy registration
# reference: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
@torch.library.custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: torch.Tensor) -> torch.Tensor:
    x_np = x.cpu().numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np).to(device=x.device)


def setup_context(ctx, inputs, output) -> torch.Tensor:
    (x,) = inputs
    ctx.save_for_backward(x)


def backward(ctx, grad):
    (x,) = ctx.saved_tensors
    return grad * x.cos()


@numpy_sin.register_fake
def _fw(x):
    return torch.empty_like(x)


torch.library.register_autograd(
    "mylib::numpy_sin", backward, setup_context=setup_context
)

global_cost = []


def default_strategy_without_cost_and_input_specs(
    op_schema: OpSchema,
) -> StrategyType:
    select_strategy = op_schema.args_schema[0]
    assert isinstance(select_strategy, OpStrategy)
    default_strategy = [
        OpSpec(
            output_specs=DTensorSpec(
                mesh=select_strategy.mesh,
                placements=strategy.output_spec.placements,
            ),
        )
        for strategy in select_strategy.strategies
    ]
    return OpStrategy(default_strategy)


def default_strategy_without_cost(
    op_schema: OpSchema, output_placement: list[Placement]
) -> StrategyType:
    global global_cost
    global_cost = []
    select_strategy = op_schema.args_schema[0]
    assert isinstance(select_strategy, OpStrategy)
    new_placement = output_placement
    output_specs = DTensorSpec(
        mesh=select_strategy.mesh,
        placements=tuple(new_placement),
        tensor_meta=select_strategy.strategies[0].output_spec.tensor_meta,
    )
    # compute the cost by do not assign to OpSpec.redistribute_cost
    cost = generate_redistribute_costs(select_strategy, output_specs)
    global_cost.append(cost)
    default_strategy = [
        OpSpec(
            output_specs=output_specs,
        )
        for _ in select_strategy.strategies
    ]
    return OpStrategy(default_strategy)


replicated_strategy_without_cost = functools.partial(
    default_strategy_without_cost, output_placement=[Replicate()]
)
sharded_strategy_without_cost = functools.partial(
    default_strategy_without_cost, output_placement=[Shard(0)]
)


def clear_strategy_cache(op_overload):
    """Clear a registered op strategy"""
    propagator = DTensor._op_dispatcher.sharding_propagator

    if op_overload in propagator.op_strategy_funcs:
        del propagator.op_strategy_funcs[op_overload]

    if op_overload in propagator.op_to_schema_info:
        del propagator.op_to_schema_info[op_overload]

    propagator.propagate_op_sharding.cache.cache_clear()


def set_op_strategy(
    op_overload, strategy_func, schema_info=None, use_register_op_strategy_wrapper=False
):
    """Set a registered op strategy"""
    if use_register_op_strategy_wrapper:
        propagator = DTensor._op_dispatcher.sharding_propagator
        propagator.register_op_strategy(op_overload, strategy_func, schema_info)
    else:
        register_op_strategy(
            op_overload, schema_info=schema_info, fill_missing_cost=True
        )(strategy_func)


class DistTensorStrategyRegistrationTest(DTensorTestBase):
    def assert_warnings_contain(self, expected_patterns, warning_list):
        """Helper function to check multiple warning patterns"""
        warning_messages = [str(w.message) for w in warning_list]
        for pattern in expected_patterns:
            found = any(re.search(pattern, msg) for msg in warning_messages)
            self.assertTrue(
                found, f"Pattern '{pattern}' not found in warnings: {warning_messages}"
            )

    @with_comms
    def test_sharding_propagation_info_generation(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        test_op = torch.ops.mylib.numpy_sin
        clear_strategy_cache(test_op.default)
        set_op_strategy(test_op.default, default_strategy_without_cost_and_input_specs)
        expected_warnings = [
            f"input_specs is not specified for {test_op.default.name()}",
            f"tensor_meta is not specified for {test_op.default.name()}",
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._test_op(
                mesh,
                lambda x: test_op(x),
                torch.randn(16),
            )
            self.assert_warnings_contain(expected_warnings, w)

    @with_comms
    @patch(
        "torch.distributed.tensor._sharding_prop.ShardingPropagator._select_strategy"
    )
    def test_distribute_cost_correctness(self, mock_select_strategy):
        costs_from__select_strategy: list[float] = []

        def mock_select_func(strategy):
            """function copied from _select_strategy but with cost capturing"""
            nonlocal costs_from__select_strategy
            if len(strategy.strategies) == 1:
                costs_from__select_strategy = strategy.strategies[0].redistribute_cost
                return strategy.strategies[0]

            op_spec_costs: list[float] = []
            for op_spec in strategy.strategies:
                assert op_spec.redistribute_cost is not None, (
                    "must set redistribute cost each OpSpec!"
                )
                redistribute_cost = sum(chain.from_iterable(op_spec.redistribute_cost))
                op_spec_costs.append(redistribute_cost)
            costs_from__select_strategy = op_spec_costs
            return strategy.strategies[op_spec_costs.index(min(op_spec_costs))]

        mock_select_strategy.side_effect = mock_select_func
        global global_cost
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        test_op = torch.ops.mylib.numpy_sin
        for strategy_func in [
            replicated_strategy_without_cost,
            sharded_strategy_without_cost,
        ]:
            clear_strategy_cache(test_op.default)
            set_op_strategy(test_op.default, strategy_func)
            input = torch.randn(16, device=self.device_type)
            input_dt = distribute_tensor(input, mesh, [Shard(0)])
            output_dt = test_op(input_dt)
            output = test_op(input)
            self.assertEqual(output_dt.full_tensor(), output)
            self.assertEqual(global_cost, costs_from__select_strategy)


if __name__ == "__main__":
    run_tests()
