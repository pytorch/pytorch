from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from ... import cutedsl_utils as cu


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
    from .scaled_grouped_mm_blockscaled import (
        _should_use_cutedsl_scaled_grouped_mm_blockscaled,
    )

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
    from .scaled_grouped_mm_blockscaled import scaled_grouped_mm_blockscaled

    return scaled_grouped_mm_blockscaled(
        mat_a, mat_b, scale_a, scale_b, cast(Tensor, offs)
    )


def _out_cond(
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
    *,
    out: object,
) -> bool:
    from .scaled_grouped_mm_blockscaled import (
        _should_use_cutedsl_scaled_grouped_mm_blockscaled_out,
    )

    return (
        _cond(
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
        and isinstance(out, Tensor)
        and _should_use_cutedsl_scaled_grouped_mm_blockscaled_out(
            cast(Tensor, mat_a), cast(Tensor, mat_b), cast(Tensor, offs), out
        )
    )


def _out_impl(
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
    *,
    out: Tensor,
) -> Tensor:
    from .scaled_grouped_mm_blockscaled import scaled_grouped_mm_blockscaled

    return scaled_grouped_mm_blockscaled(
        mat_a, mat_b, scale_a, scale_b, cast(Tensor, offs), out
    )


def register_to_dispatch() -> None:
    for op_symbol, cond, impl in (
        ("_scaled_grouped_mm_v2", _cond, _impl),
        ("_scaled_grouped_mm_v2.out", _out_cond, _out_impl),
    ):
        cu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            cond=cond,
            impl=impl,
            allow_multiple_override=True,
        )
