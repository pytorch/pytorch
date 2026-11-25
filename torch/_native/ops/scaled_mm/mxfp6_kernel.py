"""``aten::_scaled_mm_mxfp6`` through the vendored QuACK block-scaled GEMM."""

import torch
from torch._prims_common.wrappers import _maybe_resize_out
from torch._vendor.quack.blockscaled.operand import (
    BlockScaledOperand,
    MXFP6_E2M3_PACKED,
)

from ...cutedsl.hw_caps import caps


def _config(m: int, n: int, sm_count: int):
    """Pick a QuACK config by shape; eager calls cannot autotune per shape.

    QuACK's default is kept from M=512 up. Below that tiles are 128 rows, so the
    grid is ceil(N / tile_n) CTAs per 128 rows of M:

    - M <= 16: a 2-CTA cluster along N, which multicasts A; tile_n=256 when that
      grid reaches half the SMs, else 128 when its grid reaches a quarter, else 64.
    - 16 < M <= 128: the widest tile_n whose grid reaches half the SMs, else 64.
    - 128 < M < 512: 256x128 tiles on a 2-CTA cluster along M.

    Measured on GB300 (152 SMs, 2026-11-24) with cold weights for M in 16-768,
    N in 512-28672 and K in 1024-8192, against QuACK's default and the other
    tile/cluster choices above: 1.0-1.9x faster than the default.
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


def _scale_blocks(
    scale: torch.Tensor, rows: int, k: int, device: torch.device
) -> tuple[int, ...]:
    """QuACK's (ceil(rows/128), K/128, 32, 4, 4) blocks for a flat SWIZZLE_32_4_4 scale."""
    # uint8 is the e8m0 bytes, which QuACK's container also accepts.
    if scale.dtype not in (torch.float8_e8m0fnu, torch.uint8) or scale.device != device:
        raise ValueError(
            f"MXFP6 scaled_mm expected float8_e8m0fnu scales on {device}, "
            f"got {scale.dtype} on {scale.device}"
        )
    blocks = (-(-rows // 128), k // 128, 32, 4, 4)
    expected = blocks[0] * blocks[1] * 512
    if scale.numel() != expected or not scale.is_contiguous():
        raise ValueError(
            f"MXFP6 scaled_mm expected a contiguous scale of {expected} elements, "
            f"got {scale.numel()} with strides {scale.stride()}"
        )
    return blocks


def _operand(
    data: torch.Tensor, scale: torch.Tensor, blocks: tuple[int, ...], quant_dim: int
) -> BlockScaledOperand:
    """QuACK's container for one float6_e2m3fn_x4 operand, packed along K.

    ``quant_dim`` is K's dim: -1 for A [M, K/4], -2 for B [K/4, N]. The packed
    bytes become a 3K/4-byte uint8 bit stream along K.
    """
    rows_first = data if quant_dim == -1 else data.mT
    if rows_first.stride(-1) != 1:
        raise ValueError("MXFP6 scaled_mm requires operands contiguous along K")
    if rows_first.shape[0] == 1:
        # A single row's stride is arbitrary; give it the canonical one.
        rows_first = rows_first.as_strided(rows_first.shape, (rows_first.shape[1], 1))
    # TMA needs 16-byte aligned scale bases and 32-byte aligned packed 6-bit rows.
    if scale.data_ptr() % 16:
        scale = scale.clone()
    qdata = rows_first.view(torch.uint8)
    if qdata.data_ptr() % 32 or qdata.stride(0) % 32:
        raise ValueError("MXFP6 scaled_mm requires 32-byte aligned operand rows")
    if quant_dim == -2:
        qdata = qdata.mT
    return BlockScaledOperand.from_parts(
        qdata, scale.view(blocks), MXFP6_E2M3_PACKED, quant_dim=quant_dim
    )


def mxfp6_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """[M, K/4] x [K/4, N] float6_e2m3fn_x4 with one swizzled e8m0 scale each."""
    if mat_a.dtype != torch.float6_e2m3fn_x4 or mat_b.dtype != mat_a.dtype:
        raise ValueError(
            "MXFP6 scaled_mm requires float6_e2m3fn_x4 operands, got "
            f"{mat_a.dtype} and {mat_b.dtype}"
        )
    if mat_a.dim() != 2 or mat_b.dim() != 2 or mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError(
            f"MXFP6 scaled_mm requires [M, K/4] x [K/4, N] operands, got "
            f"{tuple(mat_a.shape)} and {tuple(mat_b.shape)}"
        )
    m, n, k = mat_a.shape[0], mat_b.shape[1], 4 * mat_a.shape[1]
    if k % 128:
        raise ValueError(f"MXFP6 scaled_mm requires K to be a multiple of 128, got {k}")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"MXFP6 scaled_mm requires out_dtype bfloat16 or float32, got {out_dtype}"
        )
    hw = caps(mat_a.device)
    if hw.cc[0] != 10:
        raise NotImplementedError("MXFP6 scaled_mm requires an SM100-family GPU")
    if mat_b.device != mat_a.device:
        raise ValueError(f"MXFP6 scaled_mm expected mat_b on {mat_a.device}")
    # Validate the scales before touching ``out``, even for empty products.
    blocks_a = _scale_blocks(scale_a, m, k, mat_a.device)
    blocks_b = _scale_blocks(scale_b, n, k, mat_a.device)
    if out is None:
        out = torch.empty(m, n, dtype=out_dtype, device=mat_a.device)
    else:
        if out.dtype != out_dtype or out.device != mat_a.device:
            raise ValueError(
                f"MXFP6 scaled_mm expected out to be {out_dtype} on {mat_a.device}, "
                f"got {out.dtype} on {out.device}"
            )
        _maybe_resize_out(out, (m, n))
    if m == 0 or n == 0:
        return out
    if k == 0:
        return out.zero_()
    # The containers check the format, the scale atoms and that each scale is on
    # its operand's device.
    a = _operand(mat_a, scale_a, blocks_a, quant_dim=-1)
    b = _operand(mat_b, scale_b, blocks_b, quant_dim=-2)
    # TMA stores need a 16-byte aligned output base and row stride; run into a
    # padded buffer otherwise.
    per_16_bytes = 16 // out.element_size()
    aligned = (
        out.stride(1) == 1
        and out.stride(0) % per_16_bytes == 0
        and out.data_ptr() % 16 == 0
    )
    padded_n = -(-n // per_16_bytes) * per_16_bytes
    result = (
        out
        if aligned
        else torch.empty(m, padded_n, dtype=out_dtype, device=out.device)[:, :n]
    )

    from torch._vendor.quack.gemm_interface import gemm_tuned

    gemm_tuned.fn(
        a.qdata,
        b.qdata,
        result,
        SFA=a.scale,
        SFB=b.scale,
        bs_format_a=MXFP6_E2M3_PACKED.name,
        bs_format_b=MXFP6_E2M3_PACKED.name,
        config=_config(m, n, hw.sm_count),
    )
    if result is not out:
        out.copy_(result)
    return out
