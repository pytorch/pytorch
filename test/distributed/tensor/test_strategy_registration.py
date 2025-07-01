# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import functools
import re
import warnings
from itertools import chain
from unittest.mock import patch

import numpy as np

import torch
from torch._refs import Tensor
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


# reference: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
@torch.library.custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: Tensor) -> Tensor:
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


global_cost = []


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
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        test_op = torch.ops.mylib.numpy_sin
        clear_strategy_cache(test_op.default)
        register_op_strategy(test_op.default)(
            default_strategy_without_cost_and_input_specs
        )
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
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        test_op = torch.ops.mylib.numpy_sin
        for strategy_func in [
            replicated_strategy_without_cost,
            sharded_strategy_without_cost,
        ]:
            clear_strategy_cache(test_op.default)
            register_op_strategy(test_op.default)(strategy_func)
            input = torch.randn(16, device=self.device_type)
            input_dt = distribute_tensor(input, mesh, [Shard(0)])
            output_dt = test_op(input_dt)
            output = test_op(input)
            self.assertEqual(output_dt.full_tensor(), output)
            self.assertEqual(global_cost, costs_from__select_strategy)


if __name__ == "__main__":
    run_tests()
