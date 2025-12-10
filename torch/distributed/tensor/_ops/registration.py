#  Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable
from typing import TypeAlias, TypeVar

import torch
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
    StrategyType,
)


# convenient wrapper to register sharding propagation rules
def register_prop_rule(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
) -> Callable[
    [Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]
]:
    def wrapper(
        impl: Callable[[OpSchema], OutputSharding],
    ) -> Callable[[OpSchema], OutputSharding]:
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(
                overload, impl, schema_info
            )
        return impl

    return wrapper


# Note:
# using TypeVar here allows the registration decorator to preserve the specific type info of the wrapped strategy,
# while hardcoding the typing on the wrapper (e.g. Callable[[OpSchema], StrategyType]) would mean mypy would treat
# the return value of the wrapped strategy as always being a `StrategyType` even if it were a derived class like
# MyStrategyType(StrategyType).
_OpSchemaT = TypeVar("_OpSchemaT", bound=OpSchema)
_StrategyTypeT = TypeVar("_StrategyTypeT", bound=StrategyType)
_ShardingStrategyFunc: TypeAlias = Callable[[_OpSchemaT], _StrategyTypeT]


def register_op_strategy(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
) -> Callable[[_ShardingStrategyFunc], _ShardingStrategyFunc]:
    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]

    def wrapper(impl: _ShardingStrategyFunc) -> _ShardingStrategyFunc:
        if isinstance(op, list):
            overloads = op
        else:
            overloads = [op]

        for overload in overloads:
            curr_schema_info = None
            if schema_info is None:
                specialized_args = [
                    a.name
                    for a in overload._schema.arguments
                    if a.name in arg_names_that_require_specializing_cache_strategy
                ]
                if any(specialized_args):
                    curr_schema_info = RuntimeSchemaInfo(
                        static_kwargkey=specialized_args
                    )
            else:
                curr_schema_info = schema_info
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                overload, impl, curr_schema_info
            )
        return impl

    return wrapper
