"""QuACK-backed SM120 grouped GEMM override for aten::_grouped_mm."""

from __future__ import annotations

import torch

from ... import cutedsl_utils as cu


def _valid_2d_or_3d_strides(t: torch.Tensor) -> bool:
    if t.data_ptr() % 16 != 0:
        return False
    alignment = 16 // t.element_size()
    if t.dim() == 3 and t.stride(0) % alignment != 0:
        return False
    end_dim = t.dim() - 1
    if t.stride(end_dim - 1) == 1 and t.stride(end_dim) >= max(1, t.size(end_dim - 1)):
        return t.stride(end_dim) % alignment == 0
    if t.stride(end_dim) == 1 and t.stride(end_dim - 1) >= max(1, t.size(end_dim)):
        return t.stride(end_dim - 1) % alignment == 0
    return False


def _grouped_mm_sm12x_cond(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> bool:
    if torch.version.hip is not None:
        return False
    if self.dtype not in (torch.float16, torch.bfloat16) or mat2.dtype != self.dtype:
        return False
    if out_dtype is not None and out_dtype != self.dtype:
        return False
    if offs is None:
        return False
    if offs.dim() != 1 or offs.dtype != torch.int32 or offs.stride(0) != 1:
        return False
    if bias is not None:
        return False
    if not self.is_cuda or not mat2.is_cuda or not offs.is_cuda:
        return False
    if mat2.device != self.device or offs.device != self.device:
        return False
    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    if is_cow(self) or is_cow(mat2) or is_cow(offs):
        return False
    if not _valid_2d_or_3d_strides(self) or not _valid_2d_or_3d_strides(mat2):
        return False
    major, _ = torch.cuda.get_device_capability(self.device)
    if major != 12:
        return False

    # QuACK's varlen-M GEMM maps the common grouped forward form:
    #   self: (total_m, k), mat2: (groups, k, n), offs: (groups,)
    # Its varlen-K path maps:
    #   self: (m, total_k), mat2: (total_k, n), offs: (groups,)
    if self.dim() != 2:
        return False
    if mat2.dim() == 3:
        if offs.numel() != mat2.size(0):
            return False
        if self.size(1) != mat2.size(1):
            return False
        if (
            self.size(0) == 0
            or self.size(1) == 0
            or mat2.size(0) == 0
            or mat2.size(2) == 0
        ):
            return False
        return True
    if mat2.dim() == 2:
        if self.size(1) != mat2.size(0):
            return False
        if (
            self.size(0) == 0
            or self.size(1) == 0
            or mat2.size(1) == 0
            or offs.numel() == 0
        ):
            return False
        if self.stride(0) != 1 or mat2.stride(1) != 1:
            return False
        return True
    return False


def _grouped_mm_sm12x_impl(
    self: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    from torch._vendor.quack.grouped_gemm import grouped_mm_sm12x

    if offs is None:
        raise RuntimeError("offs must be provided")
    zero = torch.zeros(1, dtype=torch.int32, device=offs.device)
    cu_seqlens = torch.cat((zero, offs))

    return grouped_mm_sm12x(self, mat2, cu_seqlens)


def register_to_dispatch() -> None:
    cu.register_op_override(
        "aten",
        "_grouped_mm",
        "CUDA",
        cond=_grouped_mm_sm12x_cond,
        impl=_grouped_mm_sm12x_impl,
    )
