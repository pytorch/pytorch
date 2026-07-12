from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor
from torch._C import (
    _ScalingType as ScalingType,  # pyrefly: ignore [missing-module-attribute]
    _SwizzleType as SwizzleType,  # pyrefly: ignore [missing-module-attribute]
)

from ... import cutedsl_utils as cu
from .scaled_grouped_mm_blockscaled import (
    _should_use_cutedsl_scaled_grouped_mm_blockscaled,
    scaled_grouped_mm_blockscaled,
)


def _cond(
    mat_a: object,
    mat_b: object,
    scale_a: object,
    scale_recipe_a: object,
    swizzle_a: object,
    scale_b: object,
    scale_recipe_b: object,
    swizzle_b: object,
    offs: object = None,
    bias: object = None,
    out_dtype: object = None,
    contraction_dim: object = (),
    use_fast_accum: object = False,
) -> bool:
    return _should_use_cutedsl_scaled_grouped_mm_blockscaled(
        mat_a,
        mat_b,
        scale_a,
        scale_recipe_a,
        swizzle_a,
        scale_b,
        scale_recipe_b,
        swizzle_b,
        offs,
        bias,
        out_dtype,
        contraction_dim,
        use_fast_accum,
    )


def _impl(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: list[Tensor],
    scale_recipe_a: list[int],
    swizzle_a: list[int],
    scale_b: list[Tensor],
    scale_recipe_b: list[int],
    swizzle_b: list[int],
    offs: Tensor | None = None,
    bias: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    contraction_dim: Sequence[int] = (),
    use_fast_accum: bool = False,
) -> Tensor:
    cutedsl_call = scaled_grouped_mm_blockscaled
    if torch.compiler.is_compiling():
        import torch._dynamo as torch_dynamo

        cutedsl_call = torch_dynamo.disable(cutedsl_call)

    return cutedsl_call(
        mat_a,
        mat_b,
        cast(list[Tensor], scale_a),
        cast(list[Tensor], scale_b),
        [ScalingType(v) for v in scale_recipe_a],
        [ScalingType(v) for v in scale_recipe_b],
        [SwizzleType(swizzle_a[0])],
        [SwizzleType(swizzle_b[0])],
        offs,
        out_dtype,
        contraction_dim,
        use_fast_accum,
        bias=bias,
    )


def register_to_dispatch() -> None:
    cu.register_op_override(
        "aten",
        "_scaled_grouped_mm_v2",
        "CUDA",
        cond=_cond,
        impl=_impl,
        allow_multiple_override=True,
    )
