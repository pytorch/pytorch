# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

from torch._ops import OpOverload
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed._tensor.ops.utils import expand_to_full_mesh_op_strategy


def register_sharding(
    op: Union[OpOverload, List[OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo] = None,
):
    def custom_strategy(
        custom_sharding_fn: Callable[
            ..., Sequence[Tuple[PlacementList, PlacementList]]
        ],
        mesh: DeviceMesh,
        op_schema: OpSchema,
    ) -> StrategyType:
        def strategy_to_spec(strategy: object) -> object:
            if isinstance(strategy, OpStrategy):
                # take the output spec from the first strategy
                return strategy.strategies[0].output_spec
            elif isinstance(strategy, TupleStrategy):
                return tuple(strategy_to_spec(s) for s in strategy.childs)
            else:
                return strategy

        args_schema = tuple(strategy_to_spec(i) for i in op_schema.args_schema)
        kwargs_schema = {
            k: strategy_to_spec(v) for k, v in op_schema.kwargs_schema.items()
        }

        acceptable_shardings = custom_sharding_fn(*args_schema, **kwargs_schema)

        single_mesh_dim_strategies: List[PlacementList] = []
        for output_specs, input_specs in acceptable_shardings:
            single_mesh_dim_strategies.append(output_specs + input_specs)

        return expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=len(op_schema.op._schema.returns),
        )

    # if user doesn't provide schema_info, we conservatively assume schema_info.static_argnum=0
    schema_info = schema_info or RuntimeSchemaInfo(0)

    def wrapper(custom_sharding_fn):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                overload, partial(custom_strategy, custom_sharding_fn), schema_info
            )

        return custom_sharding_fn

    return wrapper
