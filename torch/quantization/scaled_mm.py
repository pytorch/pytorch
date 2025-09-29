"""Functional interface for scaled_mm and friends"""

from enum import Enum
from typing import Any, Optional

import torch
from torch import Tensor


__all__ = [
    "ScalingType",
    "SwizzleType",
    "scaled_mm",
]


class ScalingType(Enum):
    Tensorwise = 0
    Rowwise = 1
    Blockwise_1x16 = 2
    Blockwise_1x32 = 3
    Blockwise_1x128 = 4
    Blockwise_128x128 = 5


class SwizzleType(Enum):
    # No swizzling
    NoSwizzle = 0
    # NVIDIA Blockwell-style swizzle
    Swizzle_32_4_4 = 1


def scaled_mm(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: Tensor | list[Tensor],
    scale_recipe_a: ScalingType | list[ScalingType],
    scale_b: Tensor | list[Tensor],
    scale_recipe_b: ScalingType | list[ScalingType],
    swizzle_a: SwizzleType | list[SwizzleType] | None = None,
    swizzle_b: SwizzleType | list[SwizzleType] | None = None,
    bias: Optional[Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.bfloat16,
    contraction_dim: Optional[list[int]] = None,
    use_fast_accum: bool = False,
    **kwargs: Any,
) -> Tensor:
    r"""
    scaled_mm(mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b, swizzle_a, swizzle_b, bias, output_dtype,
              contraction_dim, use_fast_accum)

    Applies a scaled matrix-multiply, mm(mat_a, mat_b) where the scaling of mat_a and mat_b are described by
    scale_recipe_a and scale_recipe_b respectively.

    Args:
        scale_a: Tensor containing decoding scaling factors for mat_a
        scale_recipe_a: Enum describing how mat_a has been scaled
        scale_b: Tensor containing decoding scaling factors for mat_b
        scale_recipe_b: Enum describing how mat_b has been scaled
        swizzle_a: Enum describing the swizzling pattern (if any) of scale_a
        swizzle_b: Enum describing the swizzling pattern (if any) of scale_b
        bias: optional bias term to be added to the output
        output_dtype: dtype used for the output tensor
        contraction_dim: describe which dimensions are :math:`K` in the matmul.
        use_fast_accum: enable/disable tensor-core fast accumulation (Hopper-GPUs only)
    """
    use_deprecated_api = kwargs.pop("use_deprecated_scaled_mm", False)
    if len(kwargs) > 0:
        raise RuntimeError("kwargs contains unexpected entries, ", kwargs.keys())

    if use_deprecated_api:

        def check_valid_scale_passed(
            scale: tuple[Tensor, ...] | list[Tensor] | Tensor,
        ) -> Tensor:
            if isinstance(scale, (list, tuple)):
                if len(scale) > 1:
                    raise RuntimeError(
                        "deprecated api only accepts single scales, got", len(scale)
                    )
                return scale[0]
            else:
                return scale

        scale_a_checked = check_valid_scale_passed(scale_a)
        scale_b_checked = check_valid_scale_passed(scale_b)

        return torch._scaled_mm(
            mat_a,
            mat_b,
            scale_a_checked,
            scale_b_checked,
            bias=bias,
            scale_result=None,
            out_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )

    def expand_single_value(v: Any | list[Any] | None) -> list[Any]:
        if v is None:
            return []
        elif not isinstance(v, (list)):
            return [
                v,
            ]
        else:
            return v

    scale_a = expand_single_value(scale_a)
    scale_recipe_a = expand_single_value(scale_recipe_a)
    scale_b = expand_single_value(scale_b)
    scale_recipe_b = expand_single_value(scale_recipe_b)
    swizzle_a = expand_single_value(swizzle_a)
    swizzle_b = expand_single_value(swizzle_b)

    contraction_dim = [] if not contraction_dim else contraction_dim

    # native_functions has restrictions on what can be defined
    # & passed through - std::optional<ArrayRef<Tensor>> for instance
    # *cannot* be passed, but an empty vector (list) can.
    # So, we need to convert None arguments for lists in python
    # explicitly into empty lists.
    def list_or_empty(l: list[Any] | None) -> list[Any]:
        return [] if not l else l

    def enum_list_as_int_list(l: Any | list[Any]) -> list[Any]:
        if not isinstance(l, list):
            l = [
                l,
            ]
        return [li.value for li in l]

    out = torch._scaled_mm_v2(
        mat_a,
        mat_b,
        scale_a,
        enum_list_as_int_list(scale_recipe_a),
        enum_list_as_int_list(list_or_empty(swizzle_a)),
        scale_b,
        enum_list_as_int_list(scale_recipe_b),
        enum_list_as_int_list(list_or_empty(swizzle_b)),
        bias,
        output_dtype,
        contraction_dim,
        use_fast_accum,
    )

    return out
