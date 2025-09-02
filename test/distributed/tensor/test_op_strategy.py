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
)
from torch.distributed.tensor._ops._einsum_strategy import (
    EinsumDims,
    gen_einsum_strategies,
)
from torch.distributed.tensor._ops.utils import (
    register_op_strategy,
    replicate_op_strategy,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorOpTestBase,
    DTensorTestBase,
    with_comms,
)


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
        propagator.propagate_op_sharding.cache.cache_clear()


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
    @patch(
        "torch.distributed.tensor._sharding_prop.ShardingPropagator._select_strategy"
    )
    def test_replicate_strategy_placement(self, mock_select_strategy):
        costs_from__select_strategy = []

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
                        torch.ops.c10d_functional.all_gather_into_tensor: 4,
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


if __name__ == "__main__":
    run_tests()
