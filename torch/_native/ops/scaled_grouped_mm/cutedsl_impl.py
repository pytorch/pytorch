"""CuTeDSL override for SM90 DeepSeek FP8 grouped mm."""

import torch

from ... import cutedsl_utils as cu
from ._common import any_cow
from .group_meta import allocate_output
from .hopper_deepseek_kernel import run_deepseek_grouped_gemm
from .scaled_grouped_mm_deepseek import _should_use_cutedsl_scaled_grouped_mm_deepseek


def _cond(
    self,
    mat2,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    offs=None,
    bias=None,
    out_dtype=None,
    contraction_dim=(),
    use_fast_accum=False,
) -> bool:
    return _should_use_cutedsl_scaled_grouped_mm_deepseek(
        self,
        mat2,
        scale_a,
        recipe_a,
        swizzle_a,
        scale_b,
        recipe_b,
        swizzle_b,
        offs,
        bias,
        out_dtype,
        contraction_dim,
        use_fast_accum,
    )


def _out_cond(
    self,
    mat2,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    offs=None,
    bias=None,
    out_dtype=None,
    contraction_dim=(),
    use_fast_accum=False,
    *,
    out,
) -> bool:
    if not _cond(
        self,
        mat2,
        scale_a,
        recipe_a,
        swizzle_a,
        scale_b,
        recipe_b,
        swizzle_b,
        offs,
        bias,
        out_dtype,
        contraction_dim,
        use_fast_accum,
    ):
        return False
    if any_cow(out):
        return False
    if out.device != self.device or out.data_ptr() % 16 != 0:
        return False
    want_dtype = out_dtype if out_dtype is not None else torch.bfloat16
    if out.dtype != want_dtype:
        return False
    n = mat2.size(-1)
    if out.dim() != 2 or out.size(0) != self.size(0) or out.size(1) != n:
        return False
    elem_size = 2
    alignment = max(16 // elem_size, 1)
    padded_n = -(-n // alignment) * alignment
    return out.stride() == (padded_n, 1)


def _run(
    self,
    mat2,
    scale_a,
    scale_b,
    recipe_a: int,
    recipe_b: int,
    offs,
    out_dtype,
    out=None,
) -> torch.Tensor:
    if out is None:
        out = allocate_output(
            self, mat2, out_dtype if out_dtype is not None else torch.bfloat16
        )
    device = self.get_device()
    if device == torch.cuda.current_device():
        return run_deepseek_grouped_gemm(
            self, mat2, scale_a, scale_b, recipe_a, recipe_b, offs, out
        )
    with torch.cuda.device(device):
        return run_deepseek_grouped_gemm(
            self, mat2, scale_a, scale_b, recipe_a, recipe_b, offs, out
        )


def _impl(
    self,
    mat2,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    offs=None,
    bias=None,
    out_dtype=None,
    contraction_dim=(),
    use_fast_accum=False,
) -> torch.Tensor:
    cutedsl_call = _run
    if torch.compiler.is_compiling():
        import torch._dynamo as torch_dynamo

        cutedsl_call = torch_dynamo.disable(cutedsl_call)

    return cutedsl_call(
        self,
        mat2,
        scale_a[0],
        scale_b[0],
        recipe_a[0],
        recipe_b[0],
        offs,
        out_dtype,
    )


def _out_impl(
    self,
    mat2,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    offs=None,
    bias=None,
    out_dtype=None,
    contraction_dim=(),
    use_fast_accum=False,
    *,
    out,
) -> torch.Tensor:
    cutedsl_call = _run
    if torch.compiler.is_compiling():
        import torch._dynamo as torch_dynamo

        cutedsl_call = torch_dynamo.disable(cutedsl_call)

    return cutedsl_call(
        self,
        mat2,
        scale_a[0],
        scale_b[0],
        recipe_a[0],
        recipe_b[0],
        offs,
        out_dtype,
        out=out,
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
