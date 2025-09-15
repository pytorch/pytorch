# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Sequence
from functools import partial
from typing import Callable, Union

import torch
from torch._ops import OpOverload
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy


__all__ = ["register_sharding"]


def register_sharding(op: Union[OpOverload, list[OpOverload]]):
    """
    :meth:`register_sharding` is an experimental API that allows users to register sharding
    strategies for an operator when the tensor inputs and outputs are DTensor.
    It can be useful when: (1) there doesn't exist a default sharding strategy for ``op``,
    e.g. when ``op`` is a custom operator that is not supported by :class:`DTensor`; (2)
    when users would like to overwrite default sharding strategies of existing operators.

    Args:
        op (Union[OpOverload, List[OpOverload]]):
            An op or a list of ops to register the customized sharding function.

    Returns:
        A function decorator which can be used to wrap a function that defines the sharding
        strategy for the operator specified in ``op``. The defined sharding strategy will be
        registered to DTensor and will override the default sharding strategy if DTensor has
        already implemented the operator. The customized sharding function takes the same inputs
        as the original op (except that if an arg is a :class:`torch.Tensor`, it will be
        replaced by a tensor-like object that DTensor uses internally). The function should
        return a sequence of 2-tuples, each specifying acceptable output placements and its
        corresponding intput placements.

    Example:
        >>> # xdoctest: +SKIP("distributed")
        >>> @register_sharding(aten._softmax.default)
        >>> def custom_softmax_sharding(x, dim, half_to_float):
        >>>     softmax_dim = dim if dim >= 0 else dim + x.ndim
        >>>     acceptable_shardings = []
        >>>
        >>>     all_replicate = ([Replicate()], [Replicate(), None, None])
        >>>     acceptable_shardings.append(all_replicate)
        >>>
        >>>     for sharding_dim in range(x.ndim):
        >>>         if sharding_dim != softmax_dim:
        >>>             all_sharded = (
        >>>                 [Shard(sharding_dim)],
        >>>                 [Shard(sharding_dim), None, None],
        >>>             )
        >>>             acceptable_shardings.append(all_sharded)
        >>>
        >>>     return acceptable_shardings

    .. note:: This API is currently experimental and subject to change
    """

    def custom_strategy(
        custom_sharding_fn: Callable[
            ..., Sequence[tuple[PlacementList, PlacementList]]
        ],
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

        mesh = op_schema.get_mesh_from_args()

        args_schema = tuple(strategy_to_spec(i) for i in op_schema.args_schema)
        kwargs_schema = {
            k: strategy_to_spec(v) for k, v in op_schema.kwargs_schema.items()
        }

        acceptable_shardings = custom_sharding_fn(*args_schema, **kwargs_schema)

        single_mesh_dim_strategies: list[PlacementList] = []
        for output_specs, input_specs in acceptable_shardings:
            single_mesh_dim_strategies.append(output_specs + input_specs)

        # TODO: handle out variant ops
        return expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=len(op_schema.op._schema.returns),
            inplace_op=op_schema.is_inplace_op(),
        )

    def wrapper(custom_sharding_fn):
        def derive_schema_info(op):
            # NOTE: without user directly providing RuntimeSchemaInfo, for now
            #       we create it in a conservative fashion as follows:
            #       1. let static_argnum be the first int argument
            #       2. let static_kwargkey include all the int type kwargs
            #       3. always set needs_pytree=True
            static_argnum = 100
            static_kwargkey: list[str] = []
            for i, arg in enumerate(op._schema.arguments):
                if isinstance(arg.type, torch.IntType) or (
                    isinstance(arg.type, torch.OptionalType)
                    and isinstance(arg.type.getElementType(), torch.IntType)
                ):
                    static_argnum = min(i, static_argnum)
                    if arg.kwarg_only:
                        static_kwargkey.append(arg.name)
            return RuntimeSchemaInfo(
                static_argnum, static_kwargkey or None, needs_pytree=True
            )

        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                overload,
                partial(custom_strategy, custom_sharding_fn),
                derive_schema_info(overload),
            )

        return custom_sharding_fn

    return wrapper
