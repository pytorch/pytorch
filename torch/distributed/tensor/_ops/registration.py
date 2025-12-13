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
from torch.distributed.tensor._ops.single_dim_strategy import _SingleDimStrategyFunc


def _get_registration_wrapper(
    registration_fn,
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo],
    arg_names_that_require_specializing_cache_strategy: Optional[list[str]],
):
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            curr_schema_info = None
            if (
                schema_info is None
                and arg_names_that_require_specializing_cache_strategy is not None
            ):
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
            registration_fn(overload, impl, curr_schema_info)
        return impl

    return wrapper


# convenient wrapper to register sharding propagation rules
def register_prop_rule(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
) -> Callable[
    [Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]
]:
    return _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_op_strategy,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy=None,
    )


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
    return _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_op_strategy,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy,
    )


def register_single_dim_strategy(
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo] = None,
) -> Callable[[_SingleDimStrategyFunc], _SingleDimStrategyFunc]:
    """
    Registers a single_dim_strategy function for the given op.

    A single_dim_strategy enumerates all the non-trivial sharding specifications for the operator over a single mesh
    dim. Since it generates the full set of valid shardings regardless of the given input placements, tensor inputs
    are represented via TensorMetas instead of OpStrategies like in op_strategy functions. TensorMeta inputs, along
    with all other op inputs (int, float, bool, etc) inform which sharding placements are valid. Single-dim strategies
    are fed into infra that expands them over multiple mesh dims and fills sharding placeholders with concrete sharding
    types.

    Single-dim-strategies should not list the trivial "all Replicate" rule, which is assumed for all ops.

    Sharding placeholders should be used to represent generic sharding rules in single_dim_strategies.  For example,
    most operators that support sharded inputs and outputs do not care how the shards are laid out globally, as long as
    they are consistent.  It is just as valid to run a matmul on (Shard(0), Replicate -> Shard(0)) as on
    (StridedShard(0), Replicate -> StridedShard(0)).  For this reason, we use a 'ShardingPlaceholder' to represent all
    generic sharding types.  Placeholders will be filled by concrete sharding types seen in runtime inputs, not all
    types known to DTensor.

    """
    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]
    return _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_single_dim_op_strategy,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy,
    )
