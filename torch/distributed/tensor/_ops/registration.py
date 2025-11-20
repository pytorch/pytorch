#  Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable
from typing import Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor.placement_types import Placement


_T = TypeVar("_T")
_P = ParamSpec("_P")


# convenient wrapper to register sharding propagation rules
def register_prop_rule(
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo] = None,
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


def register_op_strategy(
    op, schema_info=None
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # pyre-fixme[2]: Parameter must be annotated.

    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]

    def wrapper(impl):
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


def register_single_dim_strategy(
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo] = None,
) -> Callable[
    [Callable[[OpSchema], list[list[Placement]]]],
    Callable[[OpSchema], list[list[Placement]]],
]:
    """
    Registers a simplified op strategy that only considers a single mesh dim, taking care to expand it
    to cover all the mesh dims present in the runtime inputs.
    """
    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]

    # TODO refactor the common code out into a util
    def wrapper(impl):
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
            DTensor._op_dispatcher.sharding_propagator.register_single_dim_op_strategy(
                overload, impl, curr_schema_info
            )
        return impl

    return wrapper
