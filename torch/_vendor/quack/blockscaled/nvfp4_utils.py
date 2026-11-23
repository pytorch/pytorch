# Copyright (c) 2026, Tri Dao.
"""NVFP4 (e2m1 + e4m3 block scales) weight-GEMM utilities for SM90.

Hopper has no fp4 tensor cores, so NVFP4 weights are served as W4A16: packed
e2m1 nibbles are TMA'd to smem, decoded to bf16 in registers (scale folded),
and fed to bf16 RS-WGMMA (A from RMEM). The decode is exact: e2m1 carries 2
significand bits, the e4m3 scale 4, the product needs <= 6 < bf16's 8.

Weight layout ("blob"): the offline repack permutes packed bytes so that each
thread's 16-byte LDS per (m64, k64) tile lands values directly in WGMMA
A-fragment order. For m64k16 bf16 WGMMA, thread t = 32*w + l holds value
pairs at rows r = 16*w + l//4 and r+8, k in {2c, 2c+1, 2c+8, 2c+9} with
c = l%4; a pair is k-adjacent in one row, i.e. exactly one packed fp4 byte
and one bf16x2 register after decode.

blob[g, kt, t, j]  (uint8, shape (M/64, K/64, 128, 16)):
    b = j//4 (k16 block), p = j%4 (register pair)
    row  = 64 g + 16 (t//32) + (t%32)//4 + 8 (p%2)
    byte = 32 kt + 8 b + (t%4) + 4 (p//2)      # into (M, K/2) packed weight

SF strip: e4m3 scales repacked to (M/64, K/64, 32, 4, 2) uint8: slot
s = 8*(r//16) + r%16 (r%16 < 8) holds bytes (sf[r], sf[r+8]) per k16 block,
so a thread fetches its k-tile's scales with one 8-byte LDS at slot*8.
"""

from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op, target_version
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm

from torch._vendor.quack.cute_dsl_utils import get_compile_target_capacity

__all__ = [
    "repack_nvfp4_weight",
    "repack_nvfp4_sf",
    "dequant_nvfp4_reference",
    "decode_e2m1x8_to_bf16x8",
    "decode_u4b8x8_to_bf16x8",
    "sf_pair_to_bf16x2",
    "mul_bf16x2",
    "mul_bf16x2_bcast",
]


FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# bf16 bit patterns of the 8 e2m1 magnitudes, split into lo/hi bytes for prmt
# tables: [0, 0.5, 1, 1.5, 2, 3, 4, 6] -> 0x0000 3F00 3F80 3FC0 4000 4040 4080 40C0
_BF16_MAG_LO = [0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0]
_BF16_MAG_HI = [0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40]


def _s32(v: int) -> int:
    """Wrap an unsigned 32-bit literal into Int32 range."""
    return v - (1 << 32) if v >= (1 << 31) else v


def _pack_table(bytes8):
    lo = 0
    for i, b in enumerate(bytes8[:4]):
        lo |= b << (8 * i)
    hi = 0
    for i, b in enumerate(bytes8[4:]):
        hi |= b << (8 * i)
    return _s32(lo), _s32(hi)


LUT_LO_A, LUT_LO_B = _pack_table(_BF16_MAG_LO)
LUT_HI_A, LUT_HI_B = _pack_table(_BF16_MAG_HI)
_SIGN_MASK = _s32(0x80808080)

# e4m3 byte patterns of integer magnitudes 0..7 for the sign-magnitude int4
# (W4A8) decode: [0, 1, 2, 3, 4, 5, 6, 7]
_E4M3_MAG = [0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E]
E4M3_LUT_A, E4M3_LUT_B = _pack_table(_E4M3_MAG)


# ---------------------------------------------------------------------------
# Host-side repack
# ---------------------------------------------------------------------------


def repack_nvfp4_weight(w_packed: torch.Tensor) -> torch.Tensor:
    """(M, K/2) packed-e2m1 uint8, K-major -> (M/64, K/64, 128, 16) blob.

    M must be a multiple of 64 and K a multiple of 64 (pad beforehand;
    zero nibbles decode to 0.0 so padding is harmless).
    """
    assert w_packed.dtype == torch.uint8 and w_packed.dim() == 2
    m, kb = w_packed.shape  # kb = K/2 bytes
    assert m % 64 == 0, "M (weight rows) must be a multiple of 64"
    assert kb % 32 == 0, "K must be a multiple of 64"
    g, kt = m // 64, kb // 32
    dev = w_packed.device
    t = torch.arange(128, device=dev)
    j = torch.arange(16, device=dev)
    b, p = j >> 2, j & 3
    r16 = 16 * (t >> 5) + (t & 31) // 4  # (128,)
    c = t & 3
    row = r16[:, None] + 8 * (p & 1)[None, :]  # (128, 16)
    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]  # (128, 16)
    gg = torch.arange(g, device=dev)
    kk = torch.arange(kt, device=dev)
    rows = 64 * gg[:, None, None, None] + row[None, None]  # (g,1,128,16)
    cols = 32 * kk[None, :, None, None] + col[None, None]  # (1,kt,128,16)
    return w_packed[rows, cols].contiguous()


def repack_nvfp4_sf(sf: torch.Tensor) -> torch.Tensor:
    """(M, K/16) e4m3 scales -> (M/64, K/64, 32, 4, 2) uint8 strip."""
    assert sf.dim() == 2
    sf_u8 = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    m, nsf = sf_u8.shape
    assert m % 64 == 0 and nsf % 4 == 0
    g, kt = m // 64, nsf // 4
    dev = sf.device
    s = torch.arange(32, device=dev)
    r = 16 * (s >> 3) + (s & 7)  # (32,)
    b = torch.arange(4, device=dev)
    i = torch.arange(2, device=dev)
    rows = (
        64 * torch.arange(g, device=dev)[:, None, None, None, None]
        + r[None, None, :, None, None]
        + 8 * i[None, None, None, None, :]
    )
    cols = (
        4 * torch.arange(kt, device=dev)[None, :, None, None, None] + b[None, None, None, :, None]
    )
    return sf_u8[rows, cols].contiguous()


def repack_int4_weight(w_packed: torch.Tensor, s2f6: bool = False) -> torch.Tensor:
    """(M, K/2) packed-uint4b8 uint8 -> (M/64, K/64, 128, 16) blob for the
    magic-mantissa int4 decode: on top of the byte permutation, each 4-byte
    word (4 register pairs) is nibble-interleaved to
    [p0.lo p1.lo p2.lo p3.lo | p0.hi p1.hi p2.hi p3.hi], so pair j falls out
    of (word >> 4j) & 0x000F000F as 16-bit lanes (lo, hi). Nibbles stay raw
    uint4b8 (no sign XOR); the +8 offset folds into the decode's -136 bias.

    s2f6=True (the Blackwell hw-cvt decode, Int4 only — AWQ stays on the
    magic route): word bytes 1 and 2 swap, i.e.
    [p0.lo p1.lo p0.hi p1.hi | p2.lo p3.lo p2.hi p3.hi] — the converter
    consumes byte pairs per 16-bit half, so pair j sits in nibbles j%2 and
    j%2 + 2 of half j//2. Pass use_s2f6_int_cvt() so the blob order tracks
    the decode branch."""
    blob = repack_nvfp4_weight(w_packed)
    v = blob.view(*blob.shape[:-1], 4, 4).int()  # (..., word, byte=pair)
    lo, hi = v & 0xF, v >> 4
    bytes01 = [lo[..., 0] | (lo[..., 1] << 4), hi[..., 0] | (hi[..., 1] << 4)]
    bytes23 = [lo[..., 2] | (lo[..., 3] << 4), hi[..., 2] | (hi[..., 3] << 4)]
    if s2f6:
        order = [bytes01[0], bytes01[1], bytes23[0], bytes23[1]]
    else:
        order = [bytes01[0], bytes23[0], bytes01[1], bytes23[1]]
    b = torch.stack(order, dim=-1)
    return b.to(torch.uint8).view(blob.shape).contiguous()


def repack_int4_sf(scales: torch.Tensor, k: int, group: int = 128) -> torch.Tensor:
    """(N, K/group) bf16 group scales -> per-k-tile bf16 strip; slot
    s = 8*(r//16) + r%16 holds (sf[r], sf[r+8]) pairs.

    group % 64 == 0: one word per slot, (N/64, K/64, 32, 2), the k-tile's
    (single) group scale — tiles within a group carry duplicates.
    group == 32: two words per slot, (N/64, K/64, 32, 2, 2), word j = the
    scales of the tile's j-th 32-column group (decode selects sfw[b // 2],
    the mxfp4 word layout with bf16 pairs instead of e8m0 bytes)."""
    assert scales.dtype == torch.bfloat16
    assert group == 32 or group % 64 == 0, f"group {group} must be 32 or a multiple of 64"
    m, ng = scales.shape
    assert m % 64 == 0 and k % 64 == 0 and ng == k // group
    g, kt = m // 64, k // 64
    dev = scales.device
    ss = torch.arange(32, device=dev)
    r = 16 * (ss >> 3) + (ss & 7)
    if group == 32:
        rows = (
            64 * torch.arange(g, device=dev)[:, None, None, None, None]
            + r[None, None, :, None, None]
            + 8 * torch.arange(2, device=dev)[None, None, None, None, :]
        )
        cols = (
            2 * torch.arange(kt, device=dev)[None, :, None, None, None]
            + torch.arange(2, device=dev)[None, None, None, :, None]
        )
        return scales[rows, cols].contiguous()
    rows = (
        64 * torch.arange(g, device=dev)[:, None, None, None]
        + r[None, None, :, None]
        + 8 * torch.arange(2, device=dev)[None, None, None, :]
    )
    cols = (torch.arange(kt, device=dev) * 64 // group)[None, :, None, None]
    return scales[rows, cols].contiguous()


def repack_mxfp4_sf(sf: torch.Tensor) -> torch.Tensor:
    """(N, K/32) e8m0 scales -> (N/64, K/64, 32, 2, 2) uint8 strip: slot s
    holds bytes (sf[r](h), sf[r+8](h)) for halves h=0,1 of the k-tile."""
    sf_u8 = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    m, nsf = sf_u8.shape
    assert m % 64 == 0 and nsf % 2 == 0
    g, kt = m // 64, nsf // 2
    dev = sf.device
    ss = torch.arange(32, device=dev)
    r = 16 * (ss >> 3) + (ss & 7)
    rows = (
        64 * torch.arange(g, device=dev)[:, None, None, None, None]
        + r[None, None, :, None, None]
        + 8 * torch.arange(2, device=dev)[None, None, None, None, :]
    )
    cols = (
        2 * torch.arange(kt, device=dev)[None, :, None, None, None]
        + torch.arange(2, device=dev)[None, None, None, :, None]
    )
    return sf_u8[rows, cols].contiguous()


def quantize_int4_reference(w: torch.Tensor, group: int = 128):
    """(N, K) float -> uint4b8-packed (N, K/2) + bf16 scales (N, K/group)."""
    n, k = w.shape
    assert k % group == 0
    wb = w.float().view(n, k // group, group)
    scale = (wb.abs().amax(dim=-1) / 7.0).clamp(min=1e-8).to(torch.bfloat16)
    q = torch.clamp(torch.round(wb / scale.float()[..., None]), -8, 7).to(torch.int8)
    qb8 = (q.view(n, k) + 8).to(torch.uint8)  # uint4b8
    packed = (qb8[:, 0::2] | (qb8[:, 1::2] << 4)).contiguous()
    return packed, scale


def dequant_int4_reference(packed: torch.Tensor, scales: torch.Tensor, group: int = 128):
    n, kb = packed.shape
    lo = (packed & 0xF).float() - 8.0
    hi = (packed >> 4).float() - 8.0
    vals = torch.stack([lo, hi], dim=-1).view(n, kb * 2)
    return vals * scales.float().repeat_interleave(group, dim=1)


def quantize_int4sm_reference(w: torch.Tensor, group: int = 128):
    """(N, K) float -> sign-magnitude int4 packed (N, K/2) + bf16 scales
    (N, K/group). Nibble = sign<<3 | mag, range [-7, 7] (symmetric)."""
    n, k = w.shape
    assert k % group == 0
    wb = w.float().view(n, k // group, group)
    scale = (wb.abs().amax(dim=-1) / 7.0).clamp(min=1e-8).to(torch.bfloat16)
    q = torch.clamp(torch.round(wb / scale.float()[..., None]), -7, 7).view(n, k)
    code = (q.abs().to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)).view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale


def dequant_int4sm_reference(packed: torch.Tensor, scales: torch.Tensor, group: int = 128):
    n, kb = packed.shape
    lut = torch.tensor(
        [float(v) for v in range(8)] + [-float(v) for v in range(8)], device=packed.device
    )
    vals = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1).view(
        n, kb * 2
    )
    return vals * scales.float().repeat_interleave(group, dim=1)


def quantize_int4b8_reference(w: torch.Tensor, group: int = 128):
    """(N, K) float -> offset-binary packed (N, K/2) + bf16 scales: the same
    symmetric [-7, 7] grid as int4sm (head-to-head accuracy comparisons stay
    quantizer-identical), stored as code = q + 8 for the biased e4m3 decode.
    Dequant is dequant_int4_reference (nibble - 8 == q)."""
    n, k = w.shape
    assert k % group == 0
    wb = w.float().view(n, k // group, group)
    scale = (wb.abs().amax(dim=-1) / 7.0).clamp(min=1e-8).to(torch.bfloat16)
    q = torch.clamp(torch.round(wb / scale.float()[..., None]), -7, 7).view(n, k)
    code = (q + 8).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale


def repack_w4a8_weight(w_packed: torch.Tensor) -> torch.Tensor:
    """(M, K/2) packed sign-mag int4 -> (M/64, K/128, 128, 32) blob for the
    fp8 RS fragment: thread t = 32w+l, R_p of k32-block b holds 4 consecutive
    k at row 16w+l//4+8*(p%2), k = 32b + 4*(l%4) + 16*(p//2)."""
    assert w_packed.dtype == torch.uint8 and w_packed.dim() == 2
    m, kb = w_packed.shape
    assert m % 64 == 0 and kb % 64 == 0  # K % 128 == 0
    g, kt = m // 64, kb // 64
    dev = w_packed.device
    t = torch.arange(128, device=dev)
    j = torch.arange(32, device=dev)  # byte within thread
    b, p, e = j >> 3, (j >> 1) & 3, j & 1  # k32 block, register, byte-of-pair
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p % 2)[None, :]
    col = 16 * b[None, :] + 2 * c[:, None] + 8 * (p >> 1)[None, :] + e[None, :]
    rows = 64 * torch.arange(g, device=dev)[:, None, None, None] + row[None, None]
    cols = 64 * torch.arange(kt, device=dev)[None, :, None, None] + col[None, None]
    return w_packed[rows, cols].contiguous()


def repack_w4a8_sf(scales: torch.Tensor, k: int) -> torch.Tensor:
    """(N, K/128) bf16 group scales -> (N/64, K/128, 32, 2) strip (one scale
    per row per 128-wide k-tile; slots as in repack_int4_sf)."""
    assert scales.dtype == torch.bfloat16
    m, ng = scales.shape
    assert m % 64 == 0 and ng == k // 128
    g, kt = m // 64, k // 128
    dev = scales.device
    ss = torch.arange(32, device=dev)
    r = 16 * (ss >> 3) + (ss & 7)
    rows = (
        64 * torch.arange(g, device=dev)[:, None, None, None]
        + r[None, None, :, None]
        + 8 * torch.arange(2, device=dev)[None, None, None, :]
    )
    cols = torch.arange(kt, device=dev)[None, :, None, None]
    return scales[rows, cols].contiguous()


def fold_int4sm_scales(scales: torch.Tensor):
    """(N, K/g) bf16 group scales -> (folded bf16 scales, (N,) fp32 channel
    scales) with 7 * folded <= 448 so every scaled magnitude m * sf fits
    e4m3 (the satfinite cvt in the table build never clips)."""
    s = scales.float()
    chan = (s.amax(dim=1) * (7.0 / 448.0)).clamp(min=1e-30)
    return (s / chan[:, None]).to(torch.bfloat16), chan


def dequant_int4smf_reference(packed: torch.Tensor, sf_folded: torch.Tensor, chan: torch.Tensor):
    """Folded-W4A8 dequant: the kernel's e4m3 rounding of m * sf (single
    rounding — the fold's only loss) times the fp32 channel scale."""
    n, kb = packed.shape
    lut = torch.tensor(
        [float(v) for v in range(8)] + [-float(v) for v in range(8)], device=packed.device
    )
    vals = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1).view(
        n, kb * 2
    )
    group = kb * 2 // sf_folded.shape[1]
    scaled = (vals * sf_folded.float().repeat_interleave(group, dim=1)).to(torch.float8_e4m3fn)
    return scaled.float() * chan.float()[:, None]


def quantize_mxfp4_reference(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, K) float -> packed e2m1 (N, K/2) + e8m0 scales (N, K/32)."""
    n, k = w.shape
    assert k % 32 == 0
    wb = w.float().view(n, k // 32, 32)
    amax = wb.abs().amax(dim=-1).clamp(min=2**-126)
    # e8m0: power-of-2 scale so amax/scale <= 6 (largest e2m1)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    sf = (e + 127).to(torch.uint8)  # e8m0 biased
    q = (wb / (2.0**e)[..., None]).clamp(-6, 6)
    lut = torch.tensor(FP4_VALUES, device=w.device)
    idx = (q.unsqueeze(-1).abs() - lut).abs().argmin(dim=-1)
    code = (idx + ((q < 0) & (idx > 0)) * 8).to(torch.uint8).view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, sf


def dequant_mxfp4_reference(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    n, kb = packed.shape
    lut = torch.tensor(
        FP4_VALUES + [-v for v in FP4_VALUES], dtype=torch.float32, device=packed.device
    )
    vals = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1).view(
        n, kb * 2
    )
    scales = 2.0 ** (sf.view(torch.uint8).float() - 127.0)
    return vals * scales.repeat_interleave(32, dim=1)


def quantize_int8_reference(w: torch.Tensor):
    """(N, K) float -> int8 (N, K) + per-channel bf16 scales (N,)."""
    scale = (w.float().abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(w.float() / scale[:, None]), -127, 127).to(torch.int8)
    return q.contiguous(), scale.to(torch.bfloat16)


def repack_w8a16_weight(w_i8: torch.Tensor) -> torch.Tensor:
    """(M, K) int8 -> (M/64, K/64, 128, 32) blob for the bf16 RS fragment
    (same value map as the nibble repack, one byte per value)."""
    assert w_i8.dtype == torch.int8 and w_i8.dim() == 2
    m, k = w_i8.shape
    assert m % 64 == 0 and k % 64 == 0
    g, kt = m // 64, k // 64
    dev = w_i8.device
    t = torch.arange(128, device=dev)
    j = torch.arange(32, device=dev)
    b, p, e = j >> 3, (j >> 1) & 3, j & 1
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p % 2)[None, :]
    col = 16 * b[None, :] + 2 * c[:, None] + 8 * (p >> 1)[None, :] + e[None, :]
    rows = 64 * torch.arange(g, device=dev)[:, None, None, None] + row[None, None]
    cols = 64 * torch.arange(kt, device=dev)[None, :, None, None] + col[None, None]
    return w_i8[rows, cols].contiguous()


def quantize_fp8_reference(w: torch.Tensor):
    """(N, K) float -> e4m3 (N, K) + per-channel fp32 scales (N,)."""
    scale = (w.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (w.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.contiguous(), scale


def quantize_mxfp8_reference(w: torch.Tensor):
    """(N, K) float -> e4m3 (N, K) + e8m0 scales (N, K/32) (per-32 power-of-2
    scale so |q| <= 448)."""
    n, k = w.shape
    assert k % 32 == 0
    wb = w.float().view(n, k // 32, 32)
    amax = wb.abs().amax(dim=-1).clamp(min=2**-126)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    sf = (e + 127).to(torch.uint8)
    q = (wb / (2.0**e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(n, k).contiguous(), sf


def dequant_mxfp8_reference(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    scales = 2.0 ** (sf.view(torch.uint8).float() - 127.0)
    return q.float() * scales.repeat_interleave(32, dim=1)


def quantize_int4_awq_reference(w: torch.Tensor, group: int = 128):
    """(N, K) float -> AWQ-style asymmetric uint4 packed (N, K/2) + bf16
    scales (N, K/group) + uint4 zeros (N, K/group)."""
    n, k = w.shape
    assert k % group == 0
    wb = w.float().view(n, k // group, group)
    wmin, wmax = wb.amin(dim=-1), wb.amax(dim=-1)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
    zero = torch.clamp(torch.round(-wmin / scale), 0, 15)
    q = torch.clamp(torch.round(wb / scale[..., None]) + zero[..., None], 0, 15)
    qk = q.to(torch.uint8).view(n, k)
    packed = (qk[:, 0::2] | (qk[:, 1::2] << 4)).contiguous()
    return packed, scale.to(torch.bfloat16), zero.to(torch.uint8)


def dequant_int4_awq_reference(packed, scales, zeros, group: int = 128):
    n, kb = packed.shape
    q = torch.stack([(packed & 0xF), (packed >> 4)], dim=-1).view(n, kb * 2).float()
    s = scales.float().repeat_interleave(group, dim=1)
    z = zeros.float().repeat_interleave(group, dim=1)
    return (q - z) * s


def repack_int4_awq_sf(scales, zeros, k: int):
    """-> (N/64, K/64, 32, 2, 2) bf16 strip: slot s holds words
    ((s_r, s_r8), (c_r, c_r8)) with c = -(128 + z) — an EXACT small-integer
    bf16 constant. The kernel decode is HADD2(magic, c) = q - z exact, then
    one HMUL2 by s (single rounding). The older b = -s*(z-8) HFMA2 form
    pre-rounded b at ~ulp(8s) AND occupied both fma-pipe halves per
    instruction; the add+mul pair is exact and HMUL2 runs at ~2x HFMA2's
    issue rate."""
    assert scales.dtype == torch.bfloat16
    m, ng = scales.shape
    assert m % 64 == 0 and k % 128 == 0 and ng == k // 128
    b = (-(zeros.float() + 128.0)).to(torch.bfloat16)  # exact: integer in [-143, -128]
    g, kt = m // 64, k // 64
    dev = scales.device
    ss = torch.arange(32, device=dev)
    r = 16 * (ss >> 3) + (ss & 7)
    rows = (
        64 * torch.arange(g, device=dev)[:, None, None, None]
        + r[None, None, :, None]
        + 8 * torch.arange(2, device=dev)[None, None, None, :]
    )
    cols = (torch.arange(kt, device=dev) // 2)[None, :, None, None]
    sw = scales[rows, cols]  # (g, kt, 32, 2)
    bw = b[rows, cols]
    return torch.stack([sw, bw], dim=3).contiguous()  # (g, kt, 32, 2words, 2)


def quantize_nvfp4_reference(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference NVFP4 quantizer: (N, K) float -> (N, K/2) packed e2m1 +
    (N, K/16) e4m3 scales (amax/6 per block, round-to-nearest e2m1)."""
    n, k = w.shape
    assert k % 16 == 0
    wb = w.float().view(n, k // 16, 16)
    amax = wb.abs().amax(dim=-1).clamp(min=1e-30)
    sf = (amax / 6.0).to(torch.float8_e4m3fn)
    sf_f = sf.float().clamp(min=2**-9)
    q = (wb / sf_f[..., None]).clamp(-6, 6)
    lut = torch.tensor(FP4_VALUES, device=w.device)
    idx = (q.unsqueeze(-1).abs() - lut).abs().argmin(dim=-1)
    code = (idx + ((q < 0) & (idx > 0)) * 8).to(torch.uint8).view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, sf


def dequant_nvfp4_reference(
    w_packed: torch.Tensor, sf: torch.Tensor, tensor_scale: float = 1.0
) -> torch.Tensor:
    """Reference dequant to fp32: (M, K/2) packed + (M, K/16) e4m3 -> (M, K)."""
    m, kb = w_packed.shape
    lut = torch.tensor(
        FP4_VALUES + [-v for v in FP4_VALUES], dtype=torch.float32, device=w_packed.device
    )
    lo = (w_packed & 0xF).long()
    hi = (w_packed >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).view(m, kb * 2)
    scales = sf.view(torch.uint8).view(torch.float8_e4m3fn).float()  # (M, K/16)
    return vals * scales.repeat_interleave(16, dim=1) * tensor_scale


# ---------------------------------------------------------------------------
# Device-side decode helpers
# ---------------------------------------------------------------------------


def asm_i32(operands, asm, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        T.i32(), operands, asm, constraints, has_side_effects=False, loc=loc, ip=ip
    )


@dsl_user_op
def prmt(a: Int32, b: Int32, sel: Int32, *, loc=None, ip=None) -> Int32:
    # Not cute.arch.prmt: that wrapper emits its (identical) asm with
    # has_side_effects=True, which blocks LLVM from CSE/hoisting a pure op.
    # The nvvm.prmt dialect op is modeled Pure AND semantically known to
    # LLVM: constant selectors fold, and an immediate selector stays an
    # immediate in SASS instead of burning a register on the "=r,r,r,r"
    # constraint. prmt.b32 d, a, b, c reads {b:a} bytes selected by c,
    # which is nvvm.prmt(lo=a, hi=b, selector=c).
    res = nvvm.prmt(
        Int32(a).ir_value(loc=loc, ip=ip),
        Int32(sel).ir_value(loc=loc, ip=ip),
        nvvm.PermuteMode.DEFAULT,
        hi=Int32(b).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@cute.jit
def lop3_or_and(a: Int32, b: Int32, mask: Int32) -> Int32:
    """a | (b & mask) in one LOP3. PTX immLut convention: a -> 0xF0,
    b -> 0xCC, c -> 0xAA, so f = a | (b & c) = 0xF0 | (0xCC & 0xAA)."""
    return cute.arch.lop3(a, b, mask, 0xF0 | (0xCC & 0xAA))


@dsl_user_op
def imad_lo(a: Int32, b: int, c: int, *, loc=None, ip=None) -> Int32:
    """a * b + c (b, c compile-time immediates) via mad.lo.s32: a single
    FMA-pipe IMAD, keeping shift+bias work off the (busier) ALU pipe."""
    res = asm_i32(
        [Int32(a).ir_value(loc=loc, ip=ip)],
        f"mad.lo.s32 $0, $1, {b}, {_s32(c)};",
        "=r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@cute.jit
def lop3_and_imm_or(a: Int32, mask: int, orval: Int32) -> Int32:
    """(a & mask) | orval in one LOP3; mask is a compile-time constant (ptxas
    folds it to the LOP3 immediate form), orval a register (pass a pinned
    vreg so ptxas cannot UR-remat it)."""
    return cute.arch.lop3(a, Int32(_s32(mask)), orval, (0xF0 & 0xCC) | 0xAA)


@dsl_user_op
def add_bf16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    res = asm_i32(
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "add.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def mul_bf16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    res = asm_i32(
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def mul_bf16x2_bcast(
    a: Int32, pair: Int32, hi: cutlass.Constexpr[bool], *, loc=None, ip=None
) -> Int32:
    """a (bf16x2) times a single bf16 lane of `pair` broadcast to both lanes.
    The mov pack is folded by ptxas into an HMUL2 .H0_H0/.H1_H1 source
    swizzle, so this costs exactly one SASS instruction."""
    lane = "h" if hi else "l"
    asm = (
        "{.reg .b16 l, h; .reg .b32 bb; mov.b32 {l, h}, $2; "
        f"mov.b32 bb, {{{lane}, {lane}}}; mul.rn.bf16x2 $0, $1, bb;}}"
    )
    res = asm_i32(
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(pair).ir_value(loc=loc, ip=ip)],
        asm,
        "=r,r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def add_bf16x2_bcast(
    a: Int32, pair: Int32, hi: cutlass.Constexpr[bool], *, loc=None, ip=None
) -> Int32:
    """a (bf16x2) plus a single bf16 lane of `pair` broadcast to both lanes
    (single HADD2 with a source swizzle, like mul_bf16x2_bcast)."""
    lane = "h" if hi else "l"
    asm = (
        "{.reg .b16 l, h; .reg .b32 bb; mov.b32 {l, h}, $2; "
        f"mov.b32 bb, {{{lane}, {lane}}}; add.rn.bf16x2 $0, $1, bb;}}"
    )
    res = asm_i32(
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(pair).ir_value(loc=loc, ip=ip)],
        asm,
        "=r,r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def fma_bf16x2_bcast(
    a: Int32, s_pair: Int32, b_pair: Int32, hi: cutlass.Constexpr[bool], *, loc=None, ip=None
) -> Int32:
    """a * s_lane + b_lane with one bf16 lane of each pair broadcast (single
    HFMA2 with source swizzles)."""
    lane = "h" if hi else "l"
    asm = (
        "{.reg .b16 l, h, l2, h2; .reg .b32 ss, bb; mov.b32 {l, h}, $2; "
        f"mov.b32 ss, {{{lane}, {lane}}}; mov.b32 {{l2, h2}}, $3; "
        f"mov.b32 bb, {{{'h2' if hi else 'l2'}, {'h2' if hi else 'l2'}}}; "
        "fma.rn.bf16x2 $0, $1, ss, bb;}"
    )
    res = asm_i32(
        [
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(s_pair).ir_value(loc=loc, ip=ip),
            Int32(b_pair).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def mov_b32_vreg(v: int, *, loc=None, ip=None) -> Int32:
    """Materialize a constant in a vector register via inline asm, so LLVM
    cannot fold it into consumers (observed: ~12 IMAD.U32 R,RZ,RZ,UR per
    k-tile without this). ptxas still sees a plain mov-immediate, so under
    register pressure it may STILL demote the value to a uniform register
    and remat it UR->R at each vector consumer — when that happens (dump the
    SASS), use pin_b32_vreg instead."""
    res = llvm.inline_asm(
        T.i32(),
        [],
        f"mov.b32 $0, {v & 0xFFFFFFFF:#x};",
        "=r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def pin_b32_vreg(v: int, *, loc=None, ip=None) -> Int32:
    """Materialize a constant in a vector register laundered through %tid.x
    ((tid & 0x80000000) | v in one LOP3 — tid bit 31 is always 0, so the
    value is exact). The formal tid dependence makes the value non-uniform
    and non-rematerializable to ptxas: unlike mov_b32_vreg it survives
    register pressure (observed: 4-6 IMAD.U32 UR->R remats per decode block
    of the e2m1 LUT A-halves at the (128, 16) occupancy-2 budget). Do NOT
    use for constants that can ride an instruction's immediate slot (e.g.
    the LUT B-halves): laundering forces them into a live register."""
    res = llvm.inline_asm(
        T.i32(),
        [],
        "{.reg .b32 t; mov.u32 t, %tid.x; "
        f"lop3.b32 $0, t, 0x80000000, {v & 0xFFFFFFFF:#x}, 0xEA;}}",
        "=r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


def _arch_has_bf16_narrow_cvt():
    """True on Blackwell-family targets (sm_100/103/110/120/121, a/f): the
    PTX 9.2 direct fp4/fp8 -> bf16x2 converts (SASS F2FP.BF16.E2M1/E4M3,
    optional fused ue8m0 scale) are available. Measured on RTX 5090: the hw
    convert decodes e2m1 at 2.0x the prmt-LUT sequence's rate (2.45x with
    the fused scale) and e4m3 at 1.6x the f16-route — both are shorter AND
    faster, so the arch check is the only gate."""
    from cutlass.base_dsl.arch import Arch

    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    # Arch orders by (major, minor, suffix): everything from sm_100 up has
    # the hw cvt, and unlike the block-scaled mma there is no per-op
    # admissible list to mirror — the cvt is plain PTX.
    return arch >= Arch.sm_100


def use_s2f6_int_cvt():
    """True when the int4/int8 -> bf16 integer decodes take the Blackwell
    scaled fixed-point converter (cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2
    .s2f6x2, one F2FP.BF16.S2_6 per pair). Two gates on top of the fp4/fp8
    family's arch check: the s2f6 source type needs PTX ISA 9.1 (CUDA 13.1;
    the e2m1/e4m3 converts predate it), and the COMPILE-TARGET arch is what
    matters — under the QUACK_ARCH=120-on-H100 proxy ptxas still targets the
    physical sm_90a, so this must stay off there (get_compile_target_capacity
    exists for exactly this split). Single predicate shared by the device
    decode branch, make_i4_decode_consts, AND the host repack word order
    (repack_int4_weight(s2f6=...)) so the two sides cannot drift."""
    return get_compile_target_capacity()[0] >= 10 and target_version(min_version="13.1")


def make_decode_luts():
    """Loop-invariant vector-register LUTs for decode_e2m1x8_to_bf16x8.
    The A-halves are tid-pinned: each prmt has ONE immediate slot and ptxas
    reliably gives it to the B-half, so an unpinned A-half gets demoted to a
    uniform reg and remat-copied into the decode under tight budgets. The
    B-halves stay plain movs so ptxas CAN immediate-ize them (pinning them
    would force two extra live registers).

    Blackwell (hw-cvt decode): no LUTs — None keeps the four constants (two
    of them tid-pinned, i.e. undead-code-eliminable) out of the register
    budget."""
    if _arch_has_bf16_narrow_cvt():
        return None
    return (
        pin_b32_vreg(LUT_LO_A),
        mov_b32_vreg(LUT_LO_B),
        pin_b32_vreg(LUT_HI_A),
        mov_b32_vreg(LUT_HI_B),
    )


@dsl_user_op
def decode_e2m1x8_to_bf16x8_cvt(x: Int32, *, loc=None, ip=None):
    """8 packed e2m1 nibbles -> 4 bf16x2 via the Blackwell hw converter:
    byte j -> R_j = (bf16(v_2j), bf16(v_{2j+1})) — cvt.rn.bf16x2.e2m1x2 maps
    the low nibble to the low lane, so the nibble order matches the LUT
    sequence exactly. 4 SASS (F2FP.BF16.E2M1.UNPACK_B; the byte unpack folds
    into the converter's operand select) vs the LUT sequence's 15."""
    struct_ty = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = llvm.inline_asm(
        struct_ty,
        [Int32(x).ir_value(loc=loc, ip=ip)],
        "{.reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, $4;\n\t"
        "cvt.rn.bf16x2.e2m1x2 $0, b0;\n\t"
        "cvt.rn.bf16x2.e2m1x2 $1, b1;\n\t"
        "cvt.rn.bf16x2.e2m1x2 $2, b2;\n\t"
        "cvt.rn.bf16x2.e2m1x2 $3, b3;}",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    i32 = T.i32()
    return (
        Int32(llvm.extractvalue(i32, res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def decode_e2m1x8_mul_e8m0pair_cvt(
    x: Int32, sf_bytes: Int32, byte_idx: cutlass.Constexpr[int], *, loc=None, ip=None
):
    """8 packed e2m1 nibbles times the (row r, row r+8) ue8m0 scale bytes at
    (byte_idx, byte_idx+1) of sf_bytes -> 4 bf16x2 in fragment slot order
    (R0, R2 = row r; R1, R3 = row r+8), via the fused-scale hw converter
    (cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2). 5 SASS per 8 values (1 PRMT
    building both duplicated-byte scale operands + 4 F2FP) — replaces the
    15-op LUT decode AND the 4 HMUL2 scale multiplies AND the e8m0->bf16
    strip unpack."""
    i, j = byte_idx, byte_idx + 1
    sel = i * 0x11 + j * 0x1100  # prmt bytes [i, i, j, j]
    struct_ty = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = llvm.inline_asm(
        struct_ty,
        [Int32(x).ir_value(loc=loc, ip=ip), Int32(sf_bytes).ir_value(loc=loc, ip=ip)],
        "{.reg .b8 b0, b1, b2, b3; .reg .b32 t; .reg .b16 sr, sr8;\n\t"
        "mov.b32 {b0, b1, b2, b3}, $4;\n\t"
        f"prmt.b32 t, $5, 0, {sel:#x};\n\t"
        "mov.b32 {sr, sr8}, t;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 $0, b0, sr;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 $1, b1, sr8;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 $2, b2, sr;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 $3, b3, sr8;}",
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    i32 = T.i32()
    return (
        Int32(llvm.extractvalue(i32, res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [3], loc=loc, ip=ip)),
    )


@cute.jit
def decode_e2m1x8_to_bf16x8(x: Int32, luts) -> Tuple[Int32, Int32, Int32, Int32]:
    """Decode 8 packed e2m1 nibbles (v0 lowest) into 4 bf16x2 registers
    R_j = (bf16(v_{2j}), bf16(v_{2j+1})). luts from make_decode_luts().

    Blackwell targets take the hw converter (4 F2FP.BF16.E2M1, no LUTs —
    2.0x measured on RTX 5090); pre-Blackwell:
    15 SASS ops per 8 values (10 PRMT + 3 LOP3 + SHF + IMAD.SHL), surveyed
    tight (2026-07-27): Marlin's exponent-alignment scheme costs the same construct
    + 4 unfoldable 2^126 fixup HMUL2s (bf16's 7-bit mantissa blocks the fp16-only in-place
    high-nibble trick, and any OR-able exponent offset breaks e2m1
    subnormals); IMAD/LEA pipe moves are no-ops (fma-heavy shares dispatch with IMAD)."""
    if const_expr(_arch_has_bf16_narrow_cvt()):
        return decode_e2m1x8_to_bf16x8_cvt(x)
    lo_a, lo_b, hi_a, hi_b = luts
    # prmt reads only selector bits [15:0], so one full-width mask serves both
    # halves: the lo selector ignores the extra high bits, the hi selector is
    # just a shift of the masked word.
    xm = x & Int32(_s32(0x77777777))
    c_lo = xm
    c_hi = xm >> 16
    l03 = prmt(lo_a, lo_b, c_lo)
    h03 = prmt(hi_a, hi_b, c_lo)
    l47 = prmt(lo_a, lo_b, c_hi)
    h47 = prmt(hi_a, hi_b, c_hi)
    # sign bytes: after <<4 (even values) / as-is (odd values), the sign of
    # each value sits at bit 7 of its byte with junk below; pack the byte
    # planes unmasked, then mask+OR into the hi-byte LUT in one LOP3 each.
    se = x << 4
    s03 = prmt(se, x, Int32(0x5140))
    s47 = prmt(se, x, Int32(0x7362))
    h03 = lop3_or_and(h03, s03, Int32(_SIGN_MASK))
    h47 = lop3_or_and(h47, s47, Int32(_SIGN_MASK))
    r0 = prmt(l03, h03, Int32(0x5140))
    r1 = prmt(l03, h03, Int32(0x7362))
    r2 = prmt(l47, h47, Int32(0x5140))
    r3 = prmt(l47, h47, Int32(0x7362))
    return r0, r1, r2, r3


def make_i4_decode_consts():
    """Loop-invariant vector-register constants for decode_u4b8x8_to_bf16x8:
    (0x43004300 = bf16x2 (128, 128) exponent magic, 0xC308C308 = bf16x2
    (-136, -136) bias). tid-pinned: the magic LOP3's immediate slot is taken
    by the 0x000F000F mask and HADD2 has no bf16x2 immediate form, so both
    must be stable registers.

    Blackwell (s2f6 hw-cvt decode): no consts — everything the sequence
    needs is immediate-able (masks in LOP3 slots, the ue8m0 scale rides the
    converter as an immediate)."""
    if use_s2f6_int_cvt():
        return None
    return (pin_b32_vreg(0x43004300), pin_b32_vreg(0xC308C308))


@cute.jit
def decode_u4x8_to_magicx8(x: Int32, magic) -> Tuple[Int32, Int32, Int32, Int32]:
    """8 raw nibbles in repack_int4_weight order -> 4 bf16x2 registers in
    MAGIC form: (x >> 4j) & 0x000F000F | 0x43004300 puts pair j's nibbles in
    bf16 lanes as the exact integer 128 + nibble. 1 LOP3 (+1 SHF for j>0)
    per pair; the caller renormalizes (subtract a bias, or fold the bias
    into a per-group HADD2)."""
    return (
        lop3_and_imm_or(x, 0x000F000F, magic),
        lop3_and_imm_or(x >> 4, 0x000F000F, magic),
        lop3_and_imm_or(x >> 8, 0x000F000F, magic),
        lop3_and_imm_or(x >> 12, 0x000F000F, magic),
    )


@dsl_user_op
def decode_u4b8x8_to_bf16x8_cvt(x: Int32, *, loc=None, ip=None):
    """8 raw uint4b8 nibbles (s2f6 word order — repack_int4_weight(s2f6=True))
    -> 4 bf16x2 via the Blackwell scaled fixed-point converter. nibble^8
    staged in a byte's high nibble reads as s2f6 (= q * 16/64 = q/4) and the
    fused ue8m0 scale 0x81 (2^2) restores the exact integer q = nibble-8;
    satfinite never binds (|q| <= 8). 7 SASS per 8 values: SHL + 2 LOP3
    ((a&b)^c = 0x6A fuses the nibble mask with the sign-bit flip) + 4
    F2FP.BF16.S2_6 (the b16 half selects fold into the converter's operand
    unpack, the scale is an immediate) — vs the magic route's 11, with no
    fma-pipe HADD2s and no const registers. Output pairs come from byte
    halves — R0=(n0,n2) R1=(n1,n3) R2=(n4,n6) R3=(n5,n7) — which is why the
    s2f6 repack swaps word bytes 1 and 2."""
    struct_ty = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = llvm.inline_asm(
        struct_ty,
        [Int32(x).ir_value(loc=loc, ip=ip)],
        "{.reg .b32 e, o; .reg .b16 e0, e1, o0, o1, s;\n\t"
        "shl.b32 e, $4, 4;\n\t"
        "lop3.b32 e, e, 0xF0F0F0F0, 0x80808080, 0x6A;\n\t"
        "lop3.b32 o, $4, 0xF0F0F0F0, 0x80808080, 0x6A;\n\t"
        "mov.b32 {e0, e1}, e;\n\t"
        "mov.b32 {o0, o1}, o;\n\t"
        "mov.b16 s, 0x8181;\n\t"
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 $0, e0, s;\n\t"
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 $1, o0, s;\n\t"
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 $2, e1, s;\n\t"
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 $3, o1, s;}",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    i32 = T.i32()
    return (
        Int32(llvm.extractvalue(i32, res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [3], loc=loc, ip=ip)),
    )


@cute.jit
def decode_u4b8x8_to_bf16x8(x: Int32, consts) -> Tuple[Int32, Int32, Int32, Int32]:
    """8 raw uint4b8 nibbles in repack_int4_weight order -> 4 bf16x2 registers
    R_j = pair j, exact. Blackwell targets take the scaled s2f6 hw converter
    (7 SASS, no consts — and a matching repack word order, see
    use_s2f6_int_cvt); pre-Blackwell: magic-mantissa 128 + nibble (see
    decode_u4x8_to_magicx8); HADD2 -136 yields the exact integer q = nibble-8.
    1 LOP3 (+1 SHF for j>0) + 1 HADD2 per pair — shallower and lighter on the
    ALU pipe than the prmt/sign-extend cvt sequence."""
    if const_expr(use_s2f6_int_cvt()):
        return decode_u4b8x8_to_bf16x8_cvt(x)
    magic, bias = consts
    t0, t1, t2, t3 = decode_u4x8_to_magicx8(x, magic)
    return (
        add_bf16x2(t0, bias),
        add_bf16x2(t1, bias),
        add_bf16x2(t2, bias),
        add_bf16x2(t3, bias),
    )


def make_e4m3_luts():
    return (mov_b32_vreg(E4M3_LUT_A), mov_b32_vreg(E4M3_LUT_B))


@cute.jit
def decode_u4b8x8_to_e4m3x8_biased(x: Int32) -> Tuple[Int32, Int32]:
    """8 offset-binary nibbles (code = q+8, w4a8 repack order) -> 8 e4m3
    bytes encoding (q+8) * 2^-9 EXACTLY: e4m3 bytes 0x00-0x0F are the linear
    denormal grid v * 2^-9, so the decode is a pure unpack — 5 SASS per 8
    values (SHR + 2 AND + 2 PRMT) vs int4sm's 9, with no LUT consts. The
    codes are all positive; the +8 bias is the caller's problem (epilogue
    correction of -8 * sum(b) per k-group, cutlass PR #3432's biased
    mixed-input trick) and the 2^9 rides the promote scale."""
    lo = x & Int32(_s32(0x0F0F0F0F))
    hi = (x >> 4) & Int32(_s32(0x0F0F0F0F))
    return prmt(lo, hi, Int32(0x5140)), prmt(lo, hi, Int32(0x7362))


@cute.jit
def decode_i4smx8_to_e4m3x8(x: Int32, luts) -> Tuple[Int32, Int32]:
    """8 sign-magnitude int4 nibbles -> 8 e4m3 bytes (2 i32): magnitude via
    prmt LUT, sign planes exactly as in the e2m1 decode."""
    lut_a, lut_b = luts
    xm = x & Int32(_s32(0x77777777))
    lo = prmt(lut_a, lut_b, xm)
    hi = prmt(lut_a, lut_b, xm >> 16)
    se = x << 4
    s_lo = prmt(se, x, Int32(0x5140))
    s_hi = prmt(se, x, Int32(0x7362))
    return (
        lop3_or_and(lo, s_lo, Int32(_SIGN_MASK)),
        lop3_or_and(hi, s_hi, Int32(_SIGN_MASK)),
    )


@dsl_user_op
def e4m3x4_pack_f32(v0, v1, v2, v3, *, loc=None, ip=None) -> Int32:
    """Four f32 -> one i32 of e4m3 bytes [v0, v1, v2, v3] (v0 lowest), RN
    satfinite (cvt.rn.satfinite.e4m3x2.f32 d, a, b packs d = [b, a])."""
    from cutlass import Float32

    args = [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)]
    res = llvm.inline_asm(
        T.i32(),
        args,
        "{.reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;"
        " cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi};}",
        "=r,f,f,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@cute.jit
def build_i4sm_scaled_luts(sf: "Float32"):
    """The row's scaled magnitude tables for the folded W4A8 decode:
    (bytes [e4m3(0), e4m3(sf), e4m3(2 sf), e4m3(3 sf)],
     bytes [e4m3(4 sf) .. e4m3(7 sf)]). m * sf is exact in f32 (3-bit int x
    bf16), so each table byte is the correctly-RN-rounded product — the
    fold's single rounding."""
    from cutlass import Float32

    tbl_a = e4m3x4_pack_f32(Float32(0.0), sf, sf * 2.0, sf * 3.0)
    tbl_b = e4m3x4_pack_f32(sf * 4.0, sf * 5.0, sf * 6.0, sf * 7.0)
    return tbl_a, tbl_b


@cute.jit
def decode_i4smx8_scaled_e4m3x8(x: Int32, ta_r, tb_r, ta_r8, tb_r8) -> Tuple[Int32, Int32]:
    """decode_i4smx8_to_e4m3x8 with per-row SCALED tables (folded W4A8): the
    raw word's low nibbles are all row r and the high nibbles all row r+8
    (repack byte order), so per-row tables cost zero extra ops."""
    xm = x & Int32(_s32(0x77777777))
    lo = prmt(ta_r, tb_r, xm)
    hi = prmt(ta_r8, tb_r8, xm >> 16)
    se = x << 4
    s_lo = prmt(se, x, Int32(0x5140))
    s_hi = prmt(se, x, Int32(0x7362))
    return (
        lop3_or_and(lo, s_lo, Int32(_SIGN_MASK)),
        lop3_or_and(hi, s_hi, Int32(_SIGN_MASK)),
    )


@dsl_user_op
def i32_as_f32(v: Int32, *, loc=None, ip=None):
    """Bitcast an Int32 register to Float32 (a plain mov)."""
    res = llvm.inline_asm(
        T.f32(),
        [Int32(v).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Float32(res)


@dsl_user_op
def _dp4a_extract_f32(x: Int32, byte: int, *, loc=None, ip=None):
    """Sign-extended byte of x as f32 via dp4a (IMAD pipe) + I2F."""
    res = llvm.inline_asm(
        T.f32(),
        [Int32(x).ir_value(loc=loc, ip=ip)],
        f"{{.reg .s32 t; dp4a.s32.s32 t, $1, {1 << (8 * byte)}, 0; cvt.rn.f32.s32 $0, t;}}",
        "=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Float32(res)


@dsl_user_op
def _pack_bf16x2_f32(lo, hi, *, loc=None, ip=None) -> Int32:
    # Not cute.arch.cvt_f32x2_bf16x2: that wrapper emits cvt.rn.SATFINITE
    # (and takes a vector value); this is the plain .rn packed convert.
    res = asm_i32(
        [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@cute.jit
def decode_i8x4_to_bf16x4_dp4a(x: Int32) -> Tuple[Int32, Int32]:
    """dp4a byte-extract variant (CUTLASS/cute-dsl style): keeps the extract
    on the IMAD pipe instead of prmt, 6 instr / 4 values."""
    f0 = _dp4a_extract_f32(x, 0)
    f1 = _dp4a_extract_f32(x, 1)
    f2 = _dp4a_extract_f32(x, 2)
    f3 = _dp4a_extract_f32(x, 3)
    return _pack_bf16x2_f32(f0, f1), _pack_bf16x2_f32(f2, f3)


@dsl_user_op
def decode_i8x4_to_bf16x4_cvt(x: Int32, *, loc=None, ip=None):
    """4 int8 bytes -> 2 bf16x2 via the Blackwell scaled fixed-point
    converter: an int8 byte read as s2f6 is b/64 and the fused ue8m0 scale
    0x85 (2^6) restores it — exact, satfinite never binds. 2 SASS per 4
    values (byte pairs are the converter's natural operand, half selects
    fold) vs the dp4a route's 6. Same (bytes 0,1), (bytes 2,3) register
    order as the dp4a route — no repack change."""
    struct_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = llvm.inline_asm(
        struct_ty,
        [Int32(x).ir_value(loc=loc, ip=ip)],
        "{.reg .b16 p0, p1, s;\n\t"
        "mov.b32 {p0, p1}, $2;\n\t"
        "mov.b16 s, 0x8585;\n\t"
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 $0, p0, s;\n\t"
        "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 $1, p1, s;}",
        "=r,=r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    i32 = T.i32()
    return (
        Int32(llvm.extractvalue(i32, res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, res, [1], loc=loc, ip=ip)),
    )


@cute.jit
def decode_i8x4_to_bf16x4(x: Int32) -> Tuple[Int32, Int32]:
    """4 int8 bytes -> 2 bf16x2 registers (R0 = bytes 0,1; R1 = bytes 2,3).
    Blackwell targets take the scaled s2f6 hw converter (2 SASS);
    pre-Blackwell the dp4a extract route (6)."""
    if const_expr(use_s2f6_int_cvt()):
        return decode_i8x4_to_bf16x4_cvt(x)
    return decode_i8x4_to_bf16x4_dp4a(x)


@dsl_user_op
def _e4m3x2_to_bf16x2(
    pair: Int32, hi: cutlass.Constexpr[bool] = False, *, loc=None, ip=None
) -> Int32:
    """(e4m3 b0, e4m3 b1) in the low (hi=False) or high (hi=True) 16 bits ->
    (bf16 b0, bf16 b1). Blackwell targets take the direct hw convert
    (cvt.rn.bf16x2.e4m3x2, 1 SASS — 1.6x measured); pre-Blackwell uses the
    sm_89+ hw cvt via f16/f32 (exact for all e4m3 incl. zero/denormals;
    4 SASS: F2FP.F16.E4M3 + 2 HADD2.F32 + F2FP.BF16.F32). Either way the
    half select folds into the cvt's operand unpack (F2FP ...UNPACK_B) —
    no SHF, so hi=True beats converting `x >> 16`."""
    sel = "{_, p}" if hi else "{p, _}"
    if _arch_has_bf16_narrow_cvt():
        res = asm_i32(
            [Int32(pair).ir_value(loc=loc, ip=ip)],
            f"{{.reg .b16 p; mov.b32 {sel}, $1; cvt.rn.bf16x2.e4m3x2 $0, p;}}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
        return Int32(res)
    res = asm_i32(
        [Int32(pair).ir_value(loc=loc, ip=ip)],
        "{.reg .b16 lo, hi, p; .reg .b32 h2; .reg .f32 f0, f1; "
        f"mov.b32 {sel}, $1; cvt.rn.f16x2.e4m3x2 h2, p; "
        "mov.b32 {lo, hi}, h2; cvt.f32.f16 f0, lo; cvt.f32.f16 f1, hi; "
        "cvt.rn.bf16x2.f32 $0, f1, f0;}",
        "=r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@cute.jit
def decode_e4m3x4_to_bf16x4(x: Int32) -> Tuple[Int32, Int32]:
    """4 e4m3 bytes -> 2 bf16x2 registers (R0 = bytes 0,1; R1 = bytes 2,3)."""
    return _e4m3x2_to_bf16x2(x), _e4m3x2_to_bf16x2(x, hi=True)


@cute.jit
def sf_e8m0_pair_to_bf16x2(sf_bytes: Int32, byte_idx: cutlass.Constexpr[int]) -> Int32:
    """Two e8m0 scale bytes at (byte_idx, byte_idx+1) -> one bf16x2 register
    (sf_lo, sf_hi); pick lanes at the multiply with mul_bf16x2_bcast. e8m0 is
    a raw biased exponent, so bf16 bits = byte << 7 (exact; byte 0 maps to
    +0.0 instead of 2^-127, harmless for scales). The shift rides the FMA
    pipe as an IMAD (d*128) to stay off the busier ALU pipe."""
    sel = Int32(0x4040 + 0x0100 * (byte_idx + 1) + byte_idx)
    return imad_lo(prmt(sf_bytes, Int32(0), sel), 128, 0)


@cute.jit
def sf_pair_to_bf16x2(sf_bytes: Int32, byte_idx: cutlass.Constexpr[int]) -> Int32:
    """Two e4m3 scale bytes at (byte_idx, byte_idx+1) of sf_bytes -> one
    bf16x2 register (sf_lo, sf_hi); pick lanes at the multiply with
    mul_bf16x2_bcast (a free HMUL2 source swizzle).

    Uses the sm_89+ hw e4m3 converter via f16/f32, exact for ALL e4m3
    scales including zero, subnormals (sf = amax/6 of a tiny block IS
    subnormal below amax < 6*2^-6), and negatives.
    """
    sel = 0x4400 + 0x0010 * (byte_idx + 1) + byte_idx
    return _e4m3x2_to_bf16x2(prmt(sf_bytes, Int32(0), Int32(sel)))
