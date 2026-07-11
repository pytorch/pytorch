"""CuTeDSL override for small-M Blackwell MXFP8 ``scaled_mm``."""

import torch

from ... import cutedsl_utils as cu


_BLOCKWISE_1X32 = 3
_SWIZZLE_32_4_4 = 1
_SUPPORTED_CAPABILITIES = {(10, 0), (10, 3)}
_TMA_CAPABILITIES = {(10, 3)}
_TMA_SHAPES = {(4608, 8192), (4096, 8192)}


def _blocked_scale_numel(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * 128 * (((k // 32) + 3) // 4) * 4


def _scale_is_supported(
    scale: torch.Tensor, rows: int, k: int, device: torch.device
) -> bool:
    return bool(
        scale.device == device
        and scale.dtype == torch.float8_e8m0fnu
        and scale.ndim == 1
        and scale.is_contiguous()
        and scale.numel() == _blocked_scale_numel(rows, k)
        and (scale.storage_offset() * scale.element_size()) % 32 == 0
    )


def _tc_cond(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: list[torch.Tensor],
    recipe_a: list[int],
    swizzle_a: list[int],
    scale_b: list[torch.Tensor],
    recipe_b: list[int],
    swizzle_b: list[int],
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    contraction_dim: list[int] | tuple[int, ...] = (),
    use_fast_accum: bool = False,
) -> bool:
    """Select the transposed 128x8 tensor-core path for its exact contract."""
    if torch.version.hip is not None or not self.is_cuda or not mat2.is_cuda:
        return False
    if self.device != mat2.device:
        return False
    if torch.cuda.get_device_capability(self.device) not in _SUPPORTED_CAPABILITIES:
        return False
    if self.ndim != 2 or mat2.ndim != 2:
        return False

    m, k = self.shape
    if mat2.shape[0] != k:
        return False
    n = mat2.shape[1]
    if not 1 <= m <= 8 or n == 0 or n % 128 != 0 or k == 0 or k % 128 != 0:
        return False
    if self.dtype != torch.float8_e4m3fn or mat2.dtype != torch.float8_e4m3fn:
        return False
    if not self.is_contiguous() or self.stride() != (k, 1):
        return False
    if mat2.stride() != (1, k):
        return False
    if (self.storage_offset() * self.element_size()) % 16 != 0:
        return False
    if (mat2.storage_offset() * mat2.element_size()) % 16 != 0:
        return False

    if recipe_a != [_BLOCKWISE_1X32] or recipe_b != [_BLOCKWISE_1X32]:
        return False
    if swizzle_a != [_SWIZZLE_32_4_4] or swizzle_b != [_SWIZZLE_32_4_4]:
        return False
    if len(scale_a) != 1 or len(scale_b) != 1:
        return False
    if not _scale_is_supported(scale_a[0], m, k, self.device):
        return False
    if not _scale_is_supported(scale_b[0], n, k, self.device):
        return False

    return bias is None and out_dtype == torch.bfloat16 and len(contraction_dim) == 0


def _tc_impl(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: list[torch.Tensor],
    recipe_a: list[int],
    swizzle_a: list[int],
    scale_b: list[torch.Tensor],
    recipe_b: list[int],
    swizzle_b: list[int],
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    contraction_dim: list[int] | tuple[int, ...] = (),
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """Allocate the public result and launch the native blocked-scale kernel."""
    from .cutedsl_kernel import mxfp8_small_m_scaled_mm

    output = torch.empty(
        (self.shape[0], mat2.shape[1]), dtype=torch.bfloat16, device=self.device
    )
    return mxfp8_small_m_scaled_mm(self, mat2, scale_a[0], scale_b[0], output)


def _tma_cond(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: list[torch.Tensor],
    recipe_a: list[int],
    swizzle_a: list[int],
    scale_b: list[torch.Tensor],
    recipe_b: list[int],
    swizzle_b: list[int],
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    contraction_dim: list[int] | tuple[int, ...] = (),
    use_fast_accum: bool = False,
) -> bool:
    """Prefer the CUDA-core TMA kernel for its tuned SM103 M=1 contract."""
    if not _tc_cond(
        self,
        mat2,
        scale_a,
        recipe_a,
        swizzle_a,
        scale_b,
        recipe_b,
        swizzle_b,
        bias,
        out_dtype,
        contraction_dim,
        use_fast_accum,
    ):
        return False
    return bool(
        torch.cuda.get_device_capability(self.device) in _TMA_CAPABILITIES
        and self.shape[0] == 1
        and (mat2.shape[1], self.shape[1]) in _TMA_SHAPES
    )


def _tma_impl(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: list[torch.Tensor],
    recipe_a: list[int],
    swizzle_a: list[int],
    scale_b: list[torch.Tensor],
    recipe_b: list[int],
    swizzle_b: list[int],
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    contraction_dim: list[int] | tuple[int, ...] = (),
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """Allocate the public result and launch the M=1 TMA kernel."""
    from .cutedsl_tma_kernel import mxfp8_tma_m1_scaled_mm

    output = torch.empty((1, mat2.shape[1]), dtype=torch.bfloat16, device=self.device)
    return mxfp8_tma_m1_scaled_mm(self, mat2, scale_a[0], scale_b[0], output)


def register_to_dispatch() -> None:
    for cond, impl in ((_tma_cond, _tma_impl), (_tc_cond, _tc_impl)):
        cu.register_op_override(
            "aten",
            "_scaled_mm_v2",
            "CUDA",
            cond=cond,
            impl=impl,
            allow_multiple_override=True,
        )
