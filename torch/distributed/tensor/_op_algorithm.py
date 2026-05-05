# Copyright (c) Meta Platforms, Inc. and affiliates
from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._ops import OpOverload
    from torch.distributed.tensor._op_schema import (
        OpAlgorithm,
        OpInfo,
        OpSchema,
        OpSpec,
    )


_op_algorithm_selectors: dict[
    OpOverload, Callable[[OpSchema, OpSpec], OpAlgorithm | None]
] = {}
_op_algorithm_impls: dict[str, Callable[[OpInfo, OpAlgorithm], object]] = {}

not_implemented = object()


def register_op_algorithm_selector(
    op: OpOverload,
) -> Callable[
    [Callable[[OpSchema, OpSpec], OpAlgorithm | None]],
    Callable[[OpSchema, OpSpec], OpAlgorithm | None],
]:
    def wrapper(
        func: Callable[[OpSchema, OpSpec], OpAlgorithm | None],
    ) -> Callable[[OpSchema, OpSpec], OpAlgorithm | None]:
        _op_algorithm_selectors[op] = func
        return func

    return wrapper


def register_op_algorithm_impl(
    name: str,
) -> Callable[[Callable[[OpInfo, OpAlgorithm], object]], Callable]:
    def wrapper(func: Callable[[OpInfo, OpAlgorithm], object]) -> Callable:
        _op_algorithm_impls[name] = func
        return func

    return wrapper


def select_op_algorithm(
    op_schema: OpSchema,
    selected_strategy: OpSpec,
) -> OpAlgorithm | None:
    selector = _op_algorithm_selectors.get(op_schema.op)
    if selector is None:
        return None
    return selector(op_schema, selected_strategy)


def run_op_algorithm(op_info: OpInfo, algorithm: OpAlgorithm) -> object:
    impl = _op_algorithm_impls.get(algorithm.name)
    if impl is None:
        return not_implemented
    return impl(op_info, algorithm)
