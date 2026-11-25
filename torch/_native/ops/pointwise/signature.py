from __future__ import annotations

from functools import cache
from inspect import signature as inspect_signature
from typing import Annotated, get_args, get_origin, NamedTuple, TYPE_CHECKING

import torch
from torch._prims_common import (
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind,
    get_computation_dtype,
)
from torch._tensor_iterator import TensorIterator


if TYPE_CHECKING:
    from ...cutedsl.types import MathFn
    from ._types import DTypes, PromotionOperands


class Signature(NamedTuple):
    storage: torch.dtype
    compute: torch.dtype
    arguments: DTypes
    results: DTypes

    def input_dtype(self, index: int, cpu_scalar: bool) -> torch.dtype:
        return self.arguments[index] if cpu_scalar else self.storage


@cache
def inspect_functor(fn: MathFn) -> tuple[tuple[object, ...], tuple[object, ...]]:
    sig = inspect_signature(fn, eval_str=True)
    args = tuple(p.annotation for p in sig.parameters.values())
    result = sig.return_annotation
    returns = get_args(result) if get_origin(result) is tuple else (result,)
    return args, returns


def resolve_type(
    annotation: object, storage: torch.dtype, compute: torch.dtype
) -> torch.dtype:
    if get_origin(annotation) is Annotated:
        role = get_args(annotation)[1]
        if role == "storage":
            return storage
        if role == "compute":
            return compute
        if isinstance(role, torch.dtype):
            return role
    from ...cutedsl.dtypes import cute2torch

    if annotation not in cute2torch:
        raise TypeError(f"unsupported pointwise scalar annotation: {annotation}")
    return cute2torch[annotation]


@cache
def _native_dtypes(
    operands: PromotionOperands, promotion: PromotionKind, default_dtype: torch.dtype
) -> tuple[torch.dtype, torch.dtype]:
    # Meta operands retain the dimensioned/0D promotion tiers without CUDA allocations.
    probes = [
        torch.empty(() if zero_dim else (0,), dtype=dtype, device="meta")
        for dtype, zero_dim in operands
    ]
    if promotion is PromotionKind.BOOL_TO_LONG:
        _, dtype = elementwise_dtypes(*probes, type_promotion_kind=promotion)
        probes = [probe.to(dtype) for probe in probes]
    iterator = TensorIterator(
        outputs=[None],
        const_inputs=probes,
        promote_inputs_to_common_dtype=True,
        promote_integer_inputs_to_float=promotion is PromotionKind.INT_TO_FLOAT,
        cast_common_dtype_to_outputs=promotion is not PromotionKind.ALWAYS_BOOL,
        static_dtype=torch.bool if promotion is PromotionKind.ALWAYS_BOOL else None,
    )
    common = iterator.common_dtype
    if common is None:
        raise RuntimeError("pointwise operands have no common dtype")
    return common, iterator.output().dtype


@cache
def specialize(
    fn: MathFn,
    common: torch.dtype,
    result: torch.dtype,
    nout: int = 1,
    *,
    with_index: bool = False,
) -> Signature:
    r"""Bind arithmetic types, excluding a declared leading index."""
    args, returns = inspect_functor(fn)
    if len(returns) != nout:
        raise ValueError(f"functor returns {len(returns)} values, expected {nout}")
    opmath = get_computation_dtype(common)
    if with_index:
        if not args or resolve_type(args[0], common, opmath) is not torch.int64:
            raise TypeError("indexed functors require a leading Int64 argument")
        args = args[1:]
    arguments = tuple(resolve_type(a, common, opmath) for a in args)
    raw_results = tuple(resolve_type(r, common, opmath) for r in returns)
    results = tuple(
        result if get_origin(r) is Annotated and isinstance(get_args(r)[1], str) else d
        for r, d in zip(returns, raw_results)
    )
    # Scheduling uses native opmath width; argument types independently select casts.
    compute = max(
        (opmath,) + arguments + raw_results,
        key=lambda d: (d.itemsize, d.is_floating_point),
    )
    return Signature(common, compute, arguments, results)


class Functor(NamedTuple):
    name: str

    def signature(
        self,
        operands: PromotionOperands,
        promotion: PromotionKind = PromotionKind.DEFAULT,
        nout: int = 1,
        *,
        with_index: bool = False,
    ) -> Signature:
        from .ops import get_fn

        common, result = _native_dtypes(operands, promotion, torch.get_default_dtype())
        return specialize(
            get_fn(self.name), common, result, nout, with_index=with_index
        )
