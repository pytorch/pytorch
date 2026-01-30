# Owner(s): ["oncall: distributed"]

import itertools
import random
from contextlib import contextmanager
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
    Replicate,
    Shard,
)
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops._einsum_strategy import (
    EinsumDims,
    gen_einsum_strategies,
)
from torch.distributed.tensor._ops.utils import (
    register_op_strategy,
    replicate_op_strategy,
)
from torch.distributed.tensor.debug import _clear_sharding_prop_cache, CommDebugMode
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorOpTestBase,
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


try:
    from torch.utils._cxx_pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_leaves  # type: ignore[no-redef]


def extract_tensor_meta(t) -> TensorMeta:
    return TensorMeta(t.shape, t.stride(), t.dtype)


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

        equation = "abd,bf->abfd"  # codespell:ignore
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

    def test_bmm_diffinndim_2d_mesh(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        all_strats = gen_einsum_strategies("bmk,kn->bmn", mesh)
        self.assertEqual(len(all_strats.strategies), 25)

    def test_bmm_diffoutndim_2d_mesh(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        all_strats = gen_einsum_strategies("bmk,k->bm", mesh)
        self.assertEqual(len(all_strats.strategies), 16)

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
    @property
    def world_size(self) -> int:
        return 4

    def test_redistribute_cost_mesh_1d(self):
        mesh_1d = self.build_device_mesh()
        shard_placement = (Shard(0),)
        replica_placement = (Replicate(),)
        partial_placement = (Partial(),)

        global_tensor = torch.randn(10, 10)
        global_tensor_meta = extract_tensor_meta(global_tensor)

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

        shard0_tensor_meta = extract_tensor_meta(torch.randn(8))
        partial_tensor_meta = extract_tensor_meta(torch.randn(50, 6))
        shard1_tensor_meta = extract_tensor_meta(torch.randn(6, 8))

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
        global_tensor_meta = extract_tensor_meta(global_tensor)

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
        lhs_tensor_meta = extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = extract_tensor_meta(rhs_tensor)

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
        lhs_tensor_meta = extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = extract_tensor_meta(rhs_tensor)

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

    def test_redistribute_cost_partial_to_different_partial_is_infinite(self):
        """Test that redistributing from Partial("sum") to Partial("avg") has infinite cost.

        Converting between different Partial types (e.g., sum -> avg) is not supported,
        so the redistribute cost should be infinite to prevent this strategy from being chosen.
        """
        mesh_1d = self.build_device_mesh()
        global_tensor = torch.randn(10, 10)
        global_tensor_meta = extract_tensor_meta(global_tensor)

        partial_sum_spec = DTensorSpec(mesh_1d, (Partial("sum"),), global_tensor_meta)
        partial_avg_spec = DTensorSpec(mesh_1d, (Partial("avg"),), global_tensor_meta)

        # Cost should be infinite since converting between different Partial types is unsupported
        cost = redistribute_cost(partial_sum_spec, partial_avg_spec)
        self.assertEqual(cost, float("inf"))

    def test_redistribute_cost_with_order(self):
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )

        # Source: Shard on dim 0 across all three mesh dimensions
        source_placement = (Shard(0), Shard(0))

        # Target: Replicate on first mesh dimension, shard on others
        # This requires 2 allgathers, one on dim=0 and one on dim=1
        replicate_mesh_dim0 = (Replicate(), Shard(0))

        # Target: Replicate on second mesh dimension, shard on others
        # This requires 1 allgather on dim=1
        replicate_mesh_dim1 = (Shard(0), Replicate())

        global_tensor = torch.randn(4, 4)
        global_tensor_meta = extract_tensor_meta(global_tensor)

        source_spec = DTensorSpec(mesh_2d, source_placement, global_tensor_meta)
        target_spec_dim0 = DTensorSpec(mesh_2d, replicate_mesh_dim0, global_tensor_meta)
        target_spec_dim1 = DTensorSpec(mesh_2d, replicate_mesh_dim1, global_tensor_meta)

        # Calculate costs for allgather on each mesh dimension
        cost_mesh_dim0 = redistribute_cost(source_spec, target_spec_dim0)
        cost_mesh_dim1 = redistribute_cost(source_spec, target_spec_dim1)

        # Cost increases with earlier mesh dimensions due to the way
        # mesh dimensions are ordered (outer to inner in device hierarchy)
        self.assertGreater(cost_mesh_dim0, cost_mesh_dim1)


# -------------Test op strategy registration-------------
# custom op without List[Tensor] as input
# reference: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
@torch.library.custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    out_np = np.sin(x_np) + np.sin(y_np)
    return torch.from_numpy(out_np).to(device=x.device)


def setup_context(ctx, inputs, output):
    (x, y) = inputs
    ctx.save_for_backward(x, y)


def backward(ctx, grad):
    (x, y) = ctx.saved_tensors
    return grad * x.cos(), grad * y.cos()


@numpy_sin.register_fake
def _fw(x, y):
    return torch.empty_like(x)


torch.library.register_autograd(
    "mylib::numpy_sin", backward, setup_context=setup_context
)


# custom op with List[Tensor] as input
@torch.library.custom_op("mylib::numpy_tuple_sin", mutates_args=())
def numpy_tuple_sin(
    x: torch.Tensor, y: list[torch.Tensor], z: torch.Tensor
) -> torch.Tensor:
    x_np = x.cpu().numpy()
    y_np = [i.cpu().numpy() for i in y]
    z_np = z.cpu().numpy()

    out_np = np.sin(x_np) + np.sin(z_np) + sum(np.sin(i) for i in y_np)
    return torch.from_numpy(out_np).to(device=x.device)


def setup_tuple_context(ctx, inputs, output):
    (x, y, z) = inputs
    ctx.save_for_backward(x, y, z)


def tuple_backward(ctx, grad):
    (x, y, z) = ctx.saved_tensors
    return grad * x.cos(), [grad * i.cos() for i in y], grad * z.cos()


@numpy_tuple_sin.register_fake
def _fw_tuple(x, y, z):
    return torch.empty_like(x)


torch.library.register_autograd(
    "mylib::numpy_tuple_sin", tuple_backward, setup_context=setup_tuple_context
)


@contextmanager
def op_strategy_context(op_overload, strategy_func, schema_info=None):
    """
    Context manager for setting and clearing op strategies.
    Args:
        op_overload: The operator overload to set or clear the strategy for.
        strategy_func: The strategy function to set for the operator overload.
        schema_info: Optional schema information for the operator overload.
    Yields:
        None
    """
    propagator = DTensor._op_dispatcher.sharding_propagator
    _origin_op_strategy_funcs = None
    _origin_op_strategy_schema = None
    try:
        # register the op strategy
        if op_overload in propagator.op_strategy_funcs:
            _origin_op_strategy_funcs = propagator.op_strategy_funcs[op_overload]
            del propagator.op_strategy_funcs[op_overload]
        if op_overload in propagator.op_to_schema_info:
            _origin_op_strategy_schema = propagator.op_to_schema_info[op_overload]
            del propagator.op_to_schema_info[op_overload]
        register_op_strategy(op_overload, schema_info=schema_info)(strategy_func)
        yield
    finally:
        # clear this op strategy cache
        if _origin_op_strategy_funcs is None:
            if op_overload in propagator.op_strategy_funcs:
                del propagator.op_strategy_funcs[op_overload]
        else:
            propagator.op_strategy_funcs[op_overload] = _origin_op_strategy_funcs
        if _origin_op_strategy_schema is None:
            if op_overload in propagator.op_to_schema_info:
                del propagator.op_to_schema_info[op_overload]
        else:
            propagator.op_to_schema_info[op_overload] = _origin_op_strategy_schema
        _clear_sharding_prop_cache()


def detect_exists_identical_opspec(*args, op, mesh, strategy_function) -> bool:
    """
    Given sample input args, detect if identical OpSpecs exists under the same
    OpStrategy.

    """
    tree_args = tree_leaves(args)
    # metadata for each argument
    arg_tensor_metadata = [extract_tensor_meta(i) for i in args]
    # possible combination of placements for each arg
    arg_placement_comb = []
    for i in tree_args:
        if isinstance(i, torch.Tensor):
            # possible placement choice for argument i
            placement_choices = (Replicate(), *[Shard(i) for i in range(i.ndim)])
            # expand placement choice into full Placements for argument i
            arg_placement_comb.append(
                list(itertools.product(placement_choices, repeat=mesh.ndim))
            )
            random.shuffle(arg_placement_comb[-1])

    arg_opspec_list = []
    for idx, arg_placement in enumerate(arg_placement_comb):
        arg_opspec_list.append([])
        for placement in arg_placement:
            arg_opspec_list[idx].append(
                OpSpec(
                    output_specs=DTensorSpec(
                        mesh, placement, tensor_meta=arg_tensor_metadata[idx]
                    )
                )
            )

    op_schema = OpSchema(
        op,
        args_schema=(tuple(OpStrategy(i) for i in arg_opspec_list)),
        kwargs_schema={},
    )
    with op_strategy_context(op, strategy_function):
        output_strategy = strategy_function(op_schema)
        # OpSpec doesn't have hashing, convert to str to compare
        output_strategy_str_list = [
            str(j) for i in tree_leaves(output_strategy) for j in i.strategies
        ]
        return len(output_strategy_str_list) == len(set(output_strategy_str_list))


class DistTensorReplicateStrategyRegistrationTest(DTensorTestBase):
    @with_comms
    @patch("torch.distributed.tensor._sharding_prop._select_min_cost_strategy")
    def test_replicate_strategy_placement(self, mock_select_strategy):
        costs_from__select_strategy = []

        def mock_select_func(strategy, op_schema=None):
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
                costs_from__select_strategy.append(op_spec.redistribute_cost)
                redistribute_cost = sum(chain.from_iterable(op_spec.redistribute_cost))
                op_spec_costs.append(redistribute_cost)
            return strategy.strategies[op_spec_costs.index(min(op_spec_costs))]

        mock_select_strategy.side_effect = mock_select_func
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        comm_mode = CommDebugMode()
        test_op = torch.ops.mylib.numpy_sin
        input_x = torch.randn([8, 16, 32], device=self.device_type)
        input_y = torch.randn([8, 16, 32], device=self.device_type)
        output = test_op(input_x, input_y)
        input_x_dt = distribute_tensor(input_x, mesh, [Shard(0), Shard(1)])
        input_y_dt = distribute_tensor(input_y, mesh, [Shard(0), Shard(1)])
        x_spec = DTensorSpec(mesh, input_x_dt.placements, extract_tensor_meta(input_x))
        new_x_spec = DTensorSpec(
            mesh, (Replicate(), Replicate()), extract_tensor_meta(input_x)
        )
        y_spec = DTensorSpec(mesh, input_y_dt.placements, extract_tensor_meta(input_y))
        new_y_spec = DTensorSpec(
            mesh, (Replicate(), Replicate()), extract_tensor_meta(input_y)
        )
        with comm_mode:
            with op_strategy_context(test_op.default, replicate_op_strategy):
                output_dt = test_op(input_x_dt, input_y_dt)
                self.assertEqual(
                    comm_mode.get_comm_counts(),
                    {
                        torch.ops.c10d_functional.all_gather_into_tensor: self.world_size,
                    },
                )
                expected_cost = [
                    [redistribute_cost(x_spec, new_x_spec)],
                    [redistribute_cost(y_spec, new_y_spec)],
                ]
                self.assertEqual(expected_cost, costs_from__select_strategy)
                self.assertEqual(output_dt.full_tensor(), output)
                self.assertEqual(output_dt.placements, [Replicate(), Replicate()])
                self.assertTrue(
                    detect_exists_identical_opspec(
                        input_x,
                        input_y,
                        op=test_op.default,
                        mesh=mesh,
                        strategy_function=replicate_op_strategy,
                    )
                )

    @with_comms
    def test_tuple_replicate_strategy_placement(self):
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        test_op = torch.ops.mylib.numpy_tuple_sin
        with op_strategy_context(
            test_op.default,
            replicate_op_strategy,
            schema_info=RuntimeSchemaInfo(needs_pytree=True),
        ):
            input_x = torch.randn([8, 16, 8], device=self.device_type)
            input_y = [
                torch.randn([8, 16, 8], device=self.device_type) for _ in range(3)
            ]
            input_z = torch.randn([8, 16, 8], device=self.device_type)
            output = test_op(input_x, input_y, input_z)
            input_x_dt = distribute_tensor(input_x, mesh, [Shard(0), Shard(1)])
            input_y_dt = [
                distribute_tensor(i, mesh, [Shard(1), Shard(1)]) for i in input_y
            ]
            input_z_dt = distribute_tensor(input_z, mesh, [Shard(1), Shard(0)])
            output_dt = test_op(input_x_dt, input_y_dt, input_z_dt)
            self.assertEqual(output_dt.full_tensor(), output)
            self.assertEqual(output_dt.placements, [Replicate(), Replicate()])


class TestStrategyHashing(DTensorTestBase):
    @with_comms
    def test_call_with_different_nontensor_args(self):
        mesh = self.build_device_mesh()
        global_tensor = torch.tensor(
            [
                [29.0, 45.0, 3.0, 61.0],
                [25.0, 6.0, 21.0, 0.0],
                [1.0, 63.0, 49.0, 38.0],
                [48.0, 9.0, 55.0, 18.0],
            ]
        )
        shard_spec = [Shard(1)]
        sharded_dtensor = distribute_tensor(global_tensor, mesh, shard_spec)
        with op_strategy_context(torch.ops.aten.sort.default, replicate_op_strategy):
            # intentionally do not supply `schema_info=RuntimeSchemaInfo(1)`
            torch.sort(sharded_dtensor, dim=0)  # sort each column
            out1, _ = torch.sort(sharded_dtensor, dim=1)  # sort each row
        with op_strategy_context(torch.ops.aten.sort.default, replicate_op_strategy):
            out2, _ = torch.sort(sharded_dtensor, dim=1)
        self.assertEqual(out1.full_tensor(), out2.full_tensor())


class TestStrategyOperation(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_cache_clean(self):
        mesh = self.build_device_mesh()
        test_op = torch.ops.mylib.numpy_sin
        x = torch.randn(2, device=self.device_type)
        y = torch.randn(2, device=self.device_type)
        x_dt = distribute_tensor(x, mesh, [Shard(0)])
        y_dt = distribute_tensor(y, mesh, [Shard(0)])
        with op_strategy_context(test_op.default, replicate_op_strategy):
            self._test_op_on_dtensor(test_op, x_dt, y_dt)
        with self.assertRaisesRegex(
            NotImplementedError,
            f"Operator {test_op.default} does not have a sharding strategy registered",
        ):
            self._test_op_on_dtensor(test_op, x_dt, y_dt)


DistTensorReplicateStrategyRegistrationTestWithLocalTensor = (
    create_local_tensor_test_class(
        DistTensorReplicateStrategyRegistrationTest,
    )
)

TestStrategyHashingWithLocalTensor = create_local_tensor_test_class(
    TestStrategyHashing,
)


class TestOpSchemaMetaProperties(TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        super().tearDown()
        torch.distributed.destroy_process_group()

    def test_args_meta_mixed_opstrategy_and_tuplestrategy(self):
        """Test args_meta with both OpStrategy and TupleStrategy"""
        # Create a simple mesh
        mesh = DeviceMesh("cpu", torch.arange(4))

        # Create tensor metadata
        tensor_meta1 = TensorMeta(
            shape=torch.Size([10, 20]),
            stride=(20, 1),
            dtype=torch.float32,
        )
        tensor_meta2 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )
        tensor_meta3 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )
        tensor_meta4 = TensorMeta(
            shape=torch.Size([20, 30]),
            stride=(30, 1),
            dtype=torch.float32,
        )

        # Create OpStrategies
        op_strategy1 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(0),), tensor_meta1))]
        )
        op_strategy2 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(1),), tensor_meta2))]
        )
        op_strategy3 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Replicate(),), tensor_meta3))]
        )
        op_strategy4 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(1),), tensor_meta4))]
        )

        # Create TupleStrategy
        tuple_strategy = TupleStrategy([op_strategy2, op_strategy3])

        # Create OpSchema: (OpStrategy, TupleStrategy, OpStrategy)
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema=(op_strategy1, tuple_strategy, op_strategy4),
            kwargs_schema={},
        )

        # Test args_meta
        args_meta = op_schema.args_meta
        self.assertEqual(len(args_meta), 3)
        # First arg should be TensorMeta
        self.assertIsInstance(args_meta[0], TensorMeta)
        self.assertEqual(args_meta[0].shape, torch.Size([10, 20]))
        # Second arg should be tuple of TensorMeta
        self.assertIsInstance(args_meta[1], tuple)
        self.assertEqual(len(args_meta[1]), 2)
        self.assertIsInstance(args_meta[1][0], TensorMeta)
        self.assertIsInstance(args_meta[1][1], TensorMeta)
        self.assertEqual(args_meta[1][0].shape, torch.Size([5, 10]))
        self.assertEqual(args_meta[1][1].shape, torch.Size([5, 10]))
        # Third arg should be TensorMeta
        self.assertIsInstance(args_meta[2], TensorMeta)
        self.assertEqual(args_meta[2].shape, torch.Size([20, 30]))

    def test_kwargs_meta_mixed(self):
        """Test kwargs_meta with mixed types"""
        # Create a simple mesh
        mesh = DeviceMesh("cpu", torch.arange(4))

        # Create tensor metadata
        tensor_meta1 = TensorMeta(
            shape=torch.Size([10, 20]),
            stride=(20, 1),
            dtype=torch.float32,
        )
        tensor_meta2 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )
        tensor_meta3 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )

        # Create OpStrategies
        op_strategy1 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(0),), tensor_meta1))]
        )
        op_strategy2 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(1),), tensor_meta2))]
        )
        op_strategy3 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Replicate(),), tensor_meta3))]
        )

        # Create TupleStrategy
        tuple_strategy = TupleStrategy([op_strategy2, op_strategy3])

        # Create OpSchema with mixed kwargs
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema=(),
            kwargs_schema={
                "input": op_strategy1,
                "tensors": tuple_strategy,
                "dim": 0,
                "alpha": 1.0,
            },
        )

        # Test kwargs_meta
        kwargs_meta = op_schema.kwargs_meta
        self.assertEqual(len(kwargs_meta), 4)
        self.assertIsInstance(kwargs_meta["input"], TensorMeta)
        self.assertIsInstance(kwargs_meta["tensors"], tuple)
        self.assertEqual(len(kwargs_meta["tensors"]), 2)
        self.assertEqual(kwargs_meta["dim"], 0)
        self.assertEqual(kwargs_meta["alpha"], 1.0)


class TestExpandToFullMeshOpStrategy(TestCase):
    """Tests for expand_to_full_mesh_op_strategy function.

    These tests verify that tensor_meta is correctly assigned to input/output specs,
    especially when the placement list contains None entries (e.g., optional outputs
    like grad_bias in SDPA backward when attn_bias is not used).
    """

    def setUp(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        super().setUp()
        self.world_size = 4
        self.fake_store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=self.fake_store
        )

    def tearDown(self):
        super().tearDown()
        torch.distributed.destroy_process_group()

    def _create_op_strategy_with_tensor_meta(
        self, mesh: DeviceMesh, shape: tuple, dtype: torch.dtype
    ) -> OpStrategy:
        """Create an OpStrategy with tensor_meta for testing."""
        placements = tuple(Replicate() for _ in range(mesh.ndim))
        strides = []
        stride = 1
        for s in reversed(shape):
            strides.append(stride)
            stride *= s
        strides = tuple(reversed(strides))

        tensor_meta = TensorMeta(shape=torch.Size(shape), stride=strides, dtype=dtype)
        spec = DTensorSpec(mesh=mesh, placements=placements, tensor_meta=tensor_meta)
        op_spec = OpSpec(
            output_specs=spec, input_specs=(spec,), redistribute_cost=[[0.0]]
        )
        return OpStrategy([op_spec])

    def test_expand_strategy_with_none_in_outputs(self):
        """Test that None entries in outputs don't shift input tensor_meta assignment.

        This test simulates SDPA backward where grad_bias output is None when
        attn_bias is not used. The bug was that the None entry caused subsequent
        input specs to get wrong tensor_meta (shifted by one position).
        """
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        # Simulate MLA-style attention with different dimensions for q/k vs v
        batch, heads, seq_len = 2, 8, 64
        d_qk = 192  # query/key head dimension
        d_v = 128  # value head dimension (different!)

        # Create input OpStrategy objects with distinct shapes
        grad_out_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_v), torch.float32
        )
        query_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_qk), torch.float32
        )
        key_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_qk), torch.float32
        )
        value_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_v), torch.float32
        )

        # Build OpSchema (attn_bias is None at position 4)
        args_schema = (
            grad_out_strategy,
            query_strategy,
            key_strategy,
            value_strategy,
        )
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,  # dummy op for testing
            args_schema,
            {},
            RuntimeSchemaInfo(needs_pytree=False),
        )

        # Placement list with None at position 3 (simulating grad_bias = None)
        # Structure: [out0, out1, out2, None, in0, in1, in2, in3]
        single_mesh_dim_strategies = [
            [
                Replicate(),  # output 0
                Replicate(),  # output 1
                Replicate(),  # output 2
                None,  # output 3 (None, like grad_bias when attn_bias is None)
                Replicate(),  # input 0: grad_out
                Replicate(),  # input 1: query
                Replicate(),  # input 2: key
                Replicate(),  # input 3: value
            ]
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=4,  # 4 outputs (including the None)
        )

        # Verify the input specs have correct tensor_meta
        strategy = result.strategies[0]
        input_specs = strategy.input_specs

        self.assertEqual(len(input_specs), 4)

        # Check that each input got the correct tensor_meta shape
        # Before the fix, these would be shifted by one position
        self.assertEqual(input_specs[0].tensor_meta.shape[-1], d_v)  # grad_out
        self.assertEqual(input_specs[1].tensor_meta.shape[-1], d_qk)  # query
        self.assertEqual(input_specs[2].tensor_meta.shape[-1], d_qk)  # key
        self.assertEqual(input_specs[3].tensor_meta.shape[-1], d_v)  # value

    def test_expand_strategy_without_none_in_outputs(self):
        """Test that expand_to_full_mesh_op_strategy works correctly without None entries."""
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        # Create input OpStrategy objects
        input1_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (10, 20), torch.float32
        )
        input2_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (10, 30), torch.float32
        )

        args_schema = (input1_strategy, input2_strategy)
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema,
            {},
            RuntimeSchemaInfo(needs_pytree=False),
        )

        # Placement list without any None entries
        single_mesh_dim_strategies = [
            [
                Replicate(),  # output
                Replicate(),  # input 0
                Replicate(),  # input 1
            ]
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=1,
        )

        strategy = result.strategies[0]
        input_specs = strategy.input_specs

        self.assertEqual(len(input_specs), 2)
        self.assertEqual(input_specs[0].tensor_meta.shape, torch.Size([10, 20]))
        self.assertEqual(input_specs[1].tensor_meta.shape, torch.Size([10, 30]))

    def test_expand_strategy_with_multiple_nones_in_outputs(self):
        """Test handling of multiple None entries in outputs."""
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        input1_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (5, 10), torch.float32
        )
        input2_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (5, 20), torch.float32
        )

        args_schema = (input1_strategy, input2_strategy)
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema,
            {},
            RuntimeSchemaInfo(needs_pytree=False),
        )

        # Placement list with multiple None entries in outputs
        single_mesh_dim_strategies = [
            [
                Replicate(),  # output 0 (non-None)
                None,  # output 1 (None)
                Replicate(),  # output 2 (non-None)
                None,  # output 3 (None)
                Replicate(),  # input 0
                Replicate(),  # input 1
            ]
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=4,  # 4 outputs (2 None, 2 non-None)
        )

        strategy = result.strategies[0]
        input_specs = strategy.input_specs

        self.assertEqual(len(input_specs), 2)
        # Verify correct tensor_meta assignment despite multiple Nones
        self.assertEqual(input_specs[0].tensor_meta.shape, torch.Size([5, 10]))
        self.assertEqual(input_specs[1].tensor_meta.shape, torch.Size([5, 20]))


if __name__ == "__main__":
    run_tests()
