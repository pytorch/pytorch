import functools
from collections.abc import Sequence

import torch
from torch import Tensor


@functools.cache
def _get_cpp_scaled_grouped_mm_v2_kernel():
    from torch._native import registry

    registry.deregister_op_overrides(disable_dsl_names="cutedsl")
    try:
        get_kernel = torch._C._dispatch_get_computed_kernel_for_dispatch_key
        return get_kernel("aten::_scaled_grouped_mm_v2", "CUDA")
    finally:
        registry.reenable_op_overrides(enable_dsl_names="cutedsl")


@functools.cache
def _cuda_dispatch_keyset(device_type: str):
    dispatch_key = torch._C._dispatch_key_for_device(
        device_type
    )  # pyrefly: ignore [missing-module-attribute]
    dispatch_key = getattr(
        torch._C.DispatchKey, dispatch_key
    )  # pyrefly: ignore [missing-module-attribute]
    return torch._C.DispatchKeySet(
        dispatch_key
    )  # pyrefly: ignore [missing-module-attribute]


def _as_arg_list(value):
    return value if isinstance(value, list) else [value]


def _enum_values(value):
    return [v.value if hasattr(v, "value") else v for v in _as_arg_list(value)]


def _call_cpp_scaled_grouped_mm_v2(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a,
    scale_recipe_a,
    swizzle_a,
    scale_b,
    scale_recipe_b,
    swizzle_b,
    offs: Tensor | None = None,
    bias: Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    contraction_dim: Sequence[int] = (),
    use_fast_accum: bool = False,
) -> Tensor:
    return _get_cpp_scaled_grouped_mm_v2_kernel().call_boxed(
        _cuda_dispatch_keyset(mat_a.device.type),
        mat_a,
        mat_b,
        _as_arg_list(scale_a),
        _enum_values(scale_recipe_a),
        _enum_values(swizzle_a),
        _as_arg_list(scale_b),
        _enum_values(scale_recipe_b),
        _enum_values(swizzle_b),
        offs,
        bias,
        out_dtype,
        list(contraction_dim),
        use_fast_accum,
    )
