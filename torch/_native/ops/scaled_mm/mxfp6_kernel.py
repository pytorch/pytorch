"""``aten::_scaled_mm_mxfp6`` through the vendored QuACK block-scaled GEMM."""

import functools

import torch
from torch._prims_common.wrappers import _maybe_resize_out

from .quack_operands import quack_blockscaled_scale_view, quack_packed_operand


_FORMAT = "mxfp6_e2m3_packed"


def _unsupported(message: str) -> NotImplementedError:
    return NotImplementedError(f"MXFP6 scaled_mm {message}")


@functools.cache
def _device(device_index: int) -> tuple[int, int]:
    """(major compute capability, SM count)."""
    properties = torch.cuda.get_device_properties(device_index)
    return properties.major, properties.multi_processor_count


@functools.cache
def _config(m: int, n: int, sm_count: int):
    """Pick a QuACK config by shape.

    QuACK's default is kept from M=512 up. Below that, measured on GB300 with
    cold weights (K in {4096, 8192}, N from 4096 to 28672), tiles are 128 rows,
    so the grid is ceil(N / tile_n) CTAs per 128 rows of M:

    - M <= 16: a 2-CTA cluster along N, which multicasts A, with the widest
      tile_n whose grid reaches half the SMs for 256 or a quarter for 128,
      else 64.
    - 16 < M <= 128: the widest tile_n whose grid reaches half the SMs, else 64.
    - 128 < M < 512: 256x128 tiles on a 2-CTA cluster along M.
    """
    from torch._vendor.quack.gemm_config import (
        _blockscaled_config,
        blockscaled_default_config,
    )

    if m >= 512:
        return blockscaled_default_config(m, n)
    if m > 128:
        return _blockscaled_config(256, 128, (2, 1))
    half = sm_count // 2
    if m <= 16:
        if -(-n // 256) >= half:
            tile_n = 256
        elif -(-n // 128) >= half // 2:
            tile_n = 128
        else:
            tile_n = 64
        return _blockscaled_config(128, tile_n, (1, 2))
    tile_n = next((t for t in (256, 128) if -(-n // t) >= half), 64)
    return _blockscaled_config(128, tile_n, (1, 1))


def mxfp6_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """[M, K/4] x [K/4, N] float6_e2m3fn_x4 with one swizzled e8m0 scale each.

    The ``_scaled_mm_v2`` meta validated dtypes, recipes, layouts, K and the
    scale sizes; this adds the checks that need the device or data pointers.
    """
    major, sm_count = _device(mat_a.device.index)
    if major != 10:
        raise _unsupported("requires an SM100-family GPU")
    a = quack_packed_operand(mat_a, k_dim=-1)
    b = quack_packed_operand(mat_b, k_dim=-2)
    # TMA needs 32-byte aligned bases for packed 6-bit operands.
    if a.data_ptr() % 32 or b.data_ptr() % 32:
        raise _unsupported("requires 32-byte aligned operands")
    m, n = mat_a.shape[0], mat_b.shape[1]
    if out is None:
        out = torch.empty(m, n, dtype=out_dtype, device=mat_a.device)
    else:
        if out.dtype != out_dtype or out.device != mat_a.device:
            raise _unsupported(
                f"out must be {out_dtype} on {mat_a.device}, got {out.dtype} on {out.device}"
            )
        _maybe_resize_out(out, (m, n))
    if m == 0 or n == 0:
        return out
    # QuACK's TMA needs 16-byte aligned output rows and bases, and scale bases.
    pad = -n % (16 // out.element_size())
    result = (
        out
        if pad == 0 and out.is_contiguous() and out.data_ptr() % 16 == 0
        else torch.empty(m, n + pad, dtype=out_dtype, device=out.device)[:, :n]
    )
    scale_a, scale_b = (
        s if s.data_ptr() % 16 == 0 else s.clone() for s in (scale_a, scale_b)
    )

    from torch._vendor.quack.gemm_interface import gemm_tuned

    gemm_tuned.fn(
        a,
        b,
        result,
        SFA=quack_blockscaled_scale_view(scale_a, m, a.shape[1], _FORMAT),
        SFB=quack_blockscaled_scale_view(scale_b, n, b.shape[0], _FORMAT),
        bs_format_a=_FORMAT,
        bs_format_b=_FORMAT,
        config=_config(m, n, sm_count),
    )
    if result is not out:
        out.copy_(result)
    return out
