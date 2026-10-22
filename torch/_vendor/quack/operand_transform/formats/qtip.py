# Copyright (c) 2026, Tri Dao.
"""QTIP (arXiv 2406.11235) trellis-coded 4-bit weights for the SM90 W4A16 RS
GEMM, using the lookup-free "3INST" decode re-derived for bf16.

Bitshift trellis, re-scoped to the WGMMA fragment: one trellis sequence = one
thread's 32 values per (m64, k64) tile (2 rows x 16 k in fragment slot order),
so a lane's 16-byte LDS is a self-contained TAIL-BITING bitstream — state j
(j = 0..31) is the 16-bit window bits[4j .. 4j+16) mod 128, and decoding a k16
block touches exactly two of the thread's four raw words (funnel shift). No
codebook memory: this is QTIP's compute-based decode, which is what fits the
register-only decode_k16 contract (their shipped "HYB" mode needs a 2 KB LUT
replicated 32x in smem — a different framework feature).

Decode (bf16-native 3INST): h = s * 89226354 + 64248484 (mod 2^32), then
r = (h & 0x81FF81FF) ^ 0x3E003E00 makes each 16-bit half a bf16 with random
sign/mantissa/2-low-exponent-bits (exponent 124..127, so |half| in [0.125, 2)
and NaN/Inf are unrepresentable — any bit pattern, including TMA zero-fill,
decodes finite); value = hi + lo (add.rn.bf16x2). Same construction as the
paper's fp16 constants (their exponent window is also 2^-3..2^0), with the
mask/base moved to the bf16 fields so no f16->bf16 conversion is needed.

The trellis sequence grouping differs from the reference implementation's
16x16 weight blocks, so published QTIP checkpoints must be re-quantized (the
quantizer below); post-incoherence-processing weights are ~iid Gaussian, so
the grouping choice does not affect quantization quality — only which
sequence a weight lands in.

Per-tensor scale: values decode at codebook scale (std ~1.24); fold the
weight scale into the epilogue alpha (gemm_w4a16's tensor_scale).
"""

import math

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import T

from torch._vendor.quack.blockscaled.nvfp4_utils import asm_i32, _s32, add_bf16x2, imad_lo, mov_b32_vreg, prmt

__all__ = [
    "QTIP_MUL",
    "QTIP_ADD",
    "QTIP_FPMASK",
    "decode_qtip_k16",
    "make_qtip_consts",
    "qtip_state_values",
    "quantize_qtip_reference",
    "dequant_qtip_reference",
    "decode_qtip2_k16",
    "decode_qtip2s_k16",
    "make_qtip2_consts",
    "qtip2_state_values",
    "qtip2s_state_values",
    "quantize_qtip2_reference",
    "dequant_qtip2_reference",
]

QTIP_MUL = 89226354  # the reference 3INST LCG multiplier/increment
QTIP_ADD = 64248484
QTIP_RMASK = 0x81FF81FF  # per bf16 half: sign | 2 exponent LSBs | 7 mantissa
QTIP_FPMASK = 0x3E003E00  # exponent base 124 -> exp in {124..127}, |v| in [0.125, 2)


# ---------------------------------------------------------------------------
# Device-side decode
# ---------------------------------------------------------------------------


@dsl_user_op
def _window16(lo: Int32, hi: Int32, sh: int, *, loc=None, ip=None) -> Int32:
    """bits [sh, sh+16) of the 64-bit pair (hi:lo), zero-extended: one funnel
    shift + one AND."""
    res = asm_i32(
        [Int32(lo).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip)],
        f"{{.reg .b32 t; shf.r.clamp.b32 t, $1, $2, {sh}; and.b32 $0, t, 0xFFFF;}}",
        "=r,r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def _mad_lo(a: Int32, *, loc=None, ip=None) -> Int32:
    """a * QTIP_MUL + QTIP_ADD, mod 2^32 (one IMAD)."""
    res = asm_i32(
        [Int32(a).ir_value(loc=loc, ip=ip)],
        f"mad.lo.s32 $0, $1, {_s32(QTIP_MUL)}, {_s32(QTIP_ADD)};",
        "=r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def _and_imm_xor(a: Int32, xor_reg: Int32, *, loc=None, ip=None) -> Int32:
    """(a & QTIP_RMASK) ^ xor_reg in one LOP3 (mask immediate, base pinned in
    a vector register so ptxas cannot UR-rematerialize it in the mainloop)."""
    # immLut: a=0xF0, b=0xCC, c=0xAA -> (a & b) ^ c = 0x6A
    res = asm_i32(
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(xor_reg).ir_value(loc=loc, ip=ip)],
        f"lop3.b32 $0, $1, {_s32(QTIP_RMASK)}, $2, 0x6A;",
        "=r,r,r",
        loc=loc,
        ip=ip,
    )
    return Int32(res)


def make_qtip_consts():
    return (mov_b32_vreg(QTIP_FPMASK),)


@cute.jit
def decode_qtip_k16(xw, b, consts):
    """Decode k16 block b: states 8b+i (i = 0..7) are 16-bit windows at bit
    offsets 32b + 4i of the thread's 128-bit tail-biting stream, i.e. funnel
    shifts of (xw[b], xw[(b+1) % 4]). Returns the block's 4 packed bf16x2
    fragment registers (pair p = values 2p, 2p+1). 11 SASS per pair."""
    (fpmask,) = consts
    lo = xw[b]
    hi = xw[(b + 1) % 4]
    r0 = _decode_pair(lo, hi, 0, fpmask)
    r1 = _decode_pair(lo, hi, 8, fpmask)
    r2 = _decode_pair(lo, hi, 16, fpmask)
    r3 = _decode_pair(lo, hi, 24, fpmask)
    return r0, r1, r2, r3


@cute.jit
def _decode_pair(lo, hi, sh, fpmask):
    """One bf16x2 fragment register: states at stream bits sh, sh+4 of the
    (hi:lo) word pair."""
    r_even = _and_imm_xor(_mad_lo(_window16(lo, hi, sh)), fpmask)
    r_odd = _and_imm_xor(_mad_lo(_window16(lo, hi, sh + 4)), fpmask)
    hi_pair = prmt(r_even, r_odd, Int32(0x7632))  # (even.hi, odd.hi)
    lo_pair = prmt(r_even, r_odd, Int32(0x5410))  # (even.lo, odd.lo)
    return add_bf16x2(hi_pair, lo_pair)


# ---------------------------------------------------------------------------
# Host-side codebook / dequant (bit-exact mirror of the device decode)
# ---------------------------------------------------------------------------


def qtip_state_values(device) -> torch.Tensor:
    """(65536,) f32: decoded value of every 16-bit trellis state (the sum of
    the two bf16 halves, computed in bf16 like add.rn.bf16x2)."""
    s = torch.arange(1 << 16, device=device, dtype=torch.int64)
    h = (s * QTIP_MUL + QTIP_ADD) & 0xFFFFFFFF
    r = (h & QTIP_RMASK) ^ QTIP_FPMASK
    lo = (r & 0xFFFF).to(torch.uint16).view(torch.bfloat16)
    hi = (r >> 16).to(torch.uint16).view(torch.bfloat16)
    return (lo + hi).float()


def _seq_coords(device):
    """(row, col) within a (64, 64) tile of sequence element j (0..31) for
    thread t (0..127): fragment slot order, matching the nvfp4 blob map.
    j = 8b + 2p + e; row = 16(t//32) + (t%32)//4 + 8(p%2);
    col = 16b + 2(t%4) + 8(p//2) + e."""
    t = torch.arange(128, device=device)
    j = torch.arange(32, device=device)
    b, p, e = j >> 3, (j >> 1) & 3, j & 1
    r16 = 16 * (t >> 5) + (t & 31) // 4
    row = r16[:, None] + 8 * (p & 1)[None, :]  # (128, 32)
    col = 16 * b[None, :] + 2 * (t & 3)[:, None] + 8 * (p >> 1)[None, :] + e[None, :]
    return row, col


def _gather_sequences(w: torch.Tensor):
    """(N, K) -> (nseq, 32) f32 sequences in trellis order (+ index arrays for
    the inverse scatter)."""
    n, k = w.shape
    assert n % 64 == 0 and k % 64 == 0, "qtip requires N, K multiples of 64"
    g, kt = n // 64, k // 64
    dev = w.device
    row, col = _seq_coords(dev)
    rows = 64 * torch.arange(g, device=dev)[:, None, None, None] + row[None, None]
    cols = 64 * torch.arange(kt, device=dev)[None, :, None, None] + col[None, None]
    seqs = w[rows.expand(g, kt, 128, 32), cols.expand(g, kt, 128, 32)]
    return seqs.reshape(g * kt * 128, 32).float(), rows, cols


def _viterbi(x: torch.Tensor, values: torch.Tensor, c: torch.Tensor | None):
    """Exact Viterbi over the 2^16-state bitshift trellis for (B, 32) targets.
    Transition: s' = (s >> 4) | nib << 12, so predecessors of s' are the
    contiguous block [(s' & 0xFFF) << 4, +16) — the DP step is a minpool over
    16-groups plus a broadcast gather. c (B,) int64 or None: tail-biting
    constraint (s0 & 0xFFF == c and s31 >> 4 == c). Returns (B, 32) states."""
    bs, T = x.shape
    assert T == 32
    dev = x.device
    low12 = torch.arange(1 << 16, device=dev) & 0xFFF
    err0 = (values[None, :] - x[:, 0, None]) ** 2
    if c is None:
        cost = err0
    else:
        cost = torch.full((bs, 1 << 16), torch.inf, device=dev)
        idx0 = c[:, None] | (torch.arange(16, device=dev)[None, :] << 12)
        cost.scatter_(1, idx0, err0.gather(1, idx0))
    pre_args = []
    for j in range(1, T):
        pre_min, pre_arg = cost.view(bs, 4096, 16).min(dim=-1)
        pre_args.append(pre_arg.to(torch.uint8))
        cost = (values[None, :] - x[:, j, None]) ** 2 + pre_min[:, low12]
    states = torch.zeros(bs, T, dtype=torch.int64, device=dev)
    if c is None:
        states[:, T - 1] = cost.argmin(dim=-1)
    else:
        idx_f = (c[:, None] << 4) | torch.arange(16, device=dev)[None, :]
        states[:, T - 1] = idx_f.gather(1, cost.gather(1, idx_f).argmin(dim=-1, keepdim=True))[:, 0]
    for j in range(T - 1, 0, -1):
        q = states[:, j] & 0xFFF
        e = pre_args[j - 1].gather(1, q[:, None])[:, 0].to(torch.int64)
        states[:, j - 1] = (q << 4) | e
    return states


def _states_to_stream(states: torch.Tensor) -> torch.Tensor:
    """(B, 32) tail-biting state paths -> (B, 16) uint8 little-endian streams
    (state j = bits[4j .. 4j+16) mod 128; its low nibble contributes bits
    [4j, 4j+4))."""
    nib = (states & 0xF).to(torch.uint8)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()


def quantize_qtip_reference(w: torch.Tensor, batch_size: int = 16384):
    """(N, K) float -> ((N/64, K/64, 128, 16) uint8 stream blob, None).

    Exact 2^16-state Viterbi encode, tail-biting via the reference two-pass
    roll trick (pass 1 on the half-rolled sequence pins the 12 shared
    boundary bits, pass 2 encodes under that constraint). Offline-quality
    slow (~seconds per 10^5 sequences); values are quantized at codebook
    scale — pre-scale w and carry the scale to the GEMM epilogue."""
    n, k = w.shape
    x, _, _ = _gather_sequences(w)
    values = qtip_state_values(w.device)
    streams = []
    for i in range(math.ceil(x.shape[0] / batch_size)):
        xb = x[i * batch_size : (i + 1) * batch_size]
        s_roll = _viterbi(xb.roll(16, dims=1), values, None)
        c = s_roll[:, 16] & 0xFFF
        states = _viterbi(xb, values, c)
        assert ((states[:, 1:] & 0xFFF) == (states[:, :-1] >> 4)).all()
        assert ((states[:, 0] & 0xFFF) == (states[:, -1] >> 4)).all()
        streams.append(_states_to_stream(states))
    blob = torch.cat(streams).view(n // 64, k // 64, 128, 16)
    return blob, None


# ---------------------------------------------------------------------------
# qtip2: V=2 trellis (one 16-bit state decodes a whole bf16x2 pair) over
# T=64-value tail-biting streams (tile_k=128). Decode is 6 SASS per pair
# (prmt + IMAD.WIDE + IMAD + 2 lop3 + add.bf16x2) vs qtip's 11:
#   * the 16-bit window is byte-aligned, extracted DUPLICATED (s | s<<16) by
#     one PRMT (also covering the cross-word case); the hash consumes the
#     duplicated word, so no zero-extend AND is needed;
#   * hash = s32 * QTIP2_MUL mod 2^64 (pure multiply; IMAD.WIDE + one IMAD
#     folding the high 32 bits of the multiplier). The usual LCG addend is
#     free-ridden as per-word XOR constants inside the two mask lop3s;
#   * value pair = add.rn.bf16x2(mask(h_lo), mask(h_hi)): value 0 sums the
#     two lo bf16 lanes, value 1 the two hi lanes — no register transpose.
# The V=2 step consumes 8 bits, so tail-biting pins 8 boundary bits (vs 12)
# and T=64 halves the short-sequence tax: MSE on iid N(0,1) at K=4 is
# 0.00492 vs 0.00544 for qtip (ideal random codebook 0.00488, D-R 0.00391).
# Constants from a flat-landscape random search at T=64.
# ---------------------------------------------------------------------------

QTIP2_MUL = 0xAE859205D39C3F83
# lop3 XOR constants: exponent base bits | random draws confined to RMASK
QTIP2_X0 = QTIP_FPMASK ^ (0x50548CFA & QTIP_RMASK)
QTIP2_X1 = QTIP_FPMASK ^ (0x176322C4 & QTIP_RMASK)
# p -> prmt sel: bytes (p, p+1) of the (lo, hi) word pair, duplicated
_QTIP2_DUP_SELS = (0x1010, 0x2121, 0x3232, 0x4343)


@dsl_user_op
def _mulwide_himix(s32: Int32, *, loc=None, ip=None) -> Int64:
    """s32 * QTIP2_MUL mod 2^64 in 2 SASS (IMAD.WIDE.U32 + IMAD high fix);
    the mov.b64 pack/unpacks are register renames."""
    a_lo, a_hi = QTIP2_MUL & 0xFFFFFFFF, QTIP2_MUL >> 32
    res = llvm.inline_asm(
        T.i64(),
        [Int32(s32).ir_value(loc=loc, ip=ip)],
        "{.reg .b64 t; .reg .b32 tl, th; "
        f"mul.wide.u32 t, $1, {a_lo}; "
        "mov.b64 {tl, th}, t; "
        f"mad.lo.s32 th, $1, {_s32(a_hi)}, th; "
        "mov.b64 $0, {tl, th};}",
        "=l,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int64(res)


@dsl_user_op
def _i64_lo(h: Int64, *, loc=None, ip=None) -> Int32:
    res = llvm.inline_asm(
        T.i32(),
        [Int64(h).ir_value(loc=loc, ip=ip)],
        "{.reg .b32 t; mov.b64 {$0, t}, $1;}",
        "=r,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@dsl_user_op
def _i64_hi(h: Int64, *, loc=None, ip=None) -> Int32:
    res = llvm.inline_asm(
        T.i32(),
        [Int64(h).ir_value(loc=loc, ip=ip)],
        "{.reg .b32 t; mov.b64 {t, $0}, $1;}",
        "=r,l",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


def make_qtip2_consts():
    return (mov_b32_vreg(QTIP2_X0), mov_b32_vreg(QTIP2_X1))


# qtip2s (tile_k=64 short streams) swaps the mul.wide hash for two
# independent 32-bit LCGs of the duplicated window: IMAD.WIDE's 64-bit path
# is half-rate and its aligned-pair result cost ~13% prefill wall-clock —
# two mad.lo are the same 6 SASS/pair and pipeline cleanly (measured 1.06x
# over qtip v1 at m=4096 vs 0.87x for the wide version). MSE 0.00540 at
# T=32 (vs 0.00537 wide — noise); constants landscape flat.
QTIP2S_A1, QTIP2S_B1 = QTIP_MUL, QTIP_ADD
QTIP2S_A2, QTIP2S_B2 = 2891336453, 0x9E3779B9


@cute.jit
def decode_qtip2s_k16(xw, b, consts):
    """qtip2s: V=2 on the 16-byte / 4-word stream, 2x mad.lo hash. 6 SASS
    per pair (prmt + 2 imad + 2 lop3 + add.bf16x2)."""
    (fpmask,) = consts
    lo = xw[b]
    hi = xw[(b + 1) % 4]
    out = []
    for p in cutlass.range_constexpr(4):
        s32 = prmt(lo, hi, Int32(_QTIP2_DUP_SELS[p]))
        w0 = imad_lo(s32, QTIP2S_A1, QTIP2S_B1)
        w1 = imad_lo(s32, _s32(QTIP2S_A2), QTIP2S_B2)
        out.append(add_bf16x2(_and_imm_xor(w0, fpmask), _and_imm_xor(w1, fpmask)))
    return out[0], out[1], out[2], out[3]


@cute.jit
def decode_qtip2_k16(xw, b, consts, nw: cutlass.Constexpr[int] = 8):
    """Decode k16 block b of the 4*nw-byte tail-biting stream xw (nw words,
    8 for tile_k=128 / 4 for tile_k=64): pair p's state is bytes
    (4b+p, 4b+p+1) mod 4*nw, extracted duplicated by one PRMT. Returns the
    block's 4 packed bf16x2 fragment registers. 6 SASS per pair."""
    x0, x1 = consts
    lo = xw[b]
    hi = xw[(b + 1) % nw]
    out = []
    for p in cutlass.range_constexpr(4):
        s32 = prmt(lo, hi, Int32(_QTIP2_DUP_SELS[p]))
        h = _mulwide_himix(s32)
        out.append(add_bf16x2(_and_imm_xor(_i64_lo(h), x0), _and_imm_xor(_i64_hi(h), x1)))
    return out[0], out[1], out[2], out[3]


# ---------------------------------------------------------------------------
# qtip2 host side (bit-exact mirror of the device decode)
# ---------------------------------------------------------------------------


def _pair_values(r0: torch.Tensor, r1: torch.Tensor) -> torch.Tensor:
    """Two masked hash words -> (n, 2) f32 value pairs (per-lane bf16 sums,
    computed in bf16 like add.rn.bf16x2)."""

    def bf(x):
        return (x & 0xFFFF).to(torch.uint16).view(torch.bfloat16)

    v0 = bf(r0) + bf(r1)
    v1 = bf(r0 >> 16) + bf(r1 >> 16)
    return torch.stack([v0.float(), v1.float()], dim=1)


def qtip2_state_values(device) -> torch.Tensor:
    """(65536, 2) f32: the bf16x2 value pair of every 16-bit trellis state
    under the qtip2 (mul.wide) hash."""
    import numpy as np

    s = np.arange(1 << 16, dtype=np.uint64)
    h = (s | (s << np.uint64(16))) * np.uint64(QTIP2_MUL)
    w0 = torch.from_numpy((h & np.uint64(0xFFFFFFFF)).astype(np.int64)).to(device)
    w1 = torch.from_numpy((h >> np.uint64(32)).astype(np.int64)).to(device)
    return _pair_values((w0 & QTIP_RMASK) ^ QTIP2_X0, (w1 & QTIP_RMASK) ^ QTIP2_X1)


def qtip2s_state_values(device) -> torch.Tensor:
    """(65536, 2) f32: state values under the qtip2s (2x mad.lo) hash."""
    s = torch.arange(1 << 16, device=device, dtype=torch.int64)
    s32 = s | (s << 16)
    w0 = (s32 * QTIP2S_A1 + QTIP2S_B1) & 0xFFFFFFFF
    w1 = (s32 * QTIP2S_A2 + QTIP2S_B2) & 0xFFFFFFFF
    return _pair_values((w0 & QTIP_RMASK) ^ QTIP_FPMASK, (w1 & QTIP_RMASK) ^ QTIP_FPMASK)


def _seq_coords2(device, tile_k: int = 128):
    """(row, col) within a (64, tile_k) A-tile of sequence value j
    (0..tile_k//2-1) for thread t (0..127): fragment slot order over the
    tile_k//16 k16 blocks (same map as _seq_coords with b extended)."""
    nval = tile_k // 2
    t = torch.arange(128, device=device)
    j = torch.arange(nval, device=device)
    b, p, e = j >> 3, (j >> 1) & 3, j & 1
    r16 = 16 * (t >> 5) + (t & 31) // 4
    row = r16[:, None] + 8 * (p & 1)[None, :]  # (128, nval)
    col = 16 * b[None, :] + 2 * (t & 3)[:, None] + 8 * (p >> 1)[None, :] + e[None, :]
    return row, col


def _gather_sequences2(w: torch.Tensor, tile_k: int):
    """(N, K) -> (nseq, tile_k // 2) f32 sequences in trellis value order."""
    n, k = w.shape
    assert n % 64 == 0 and k % tile_k == 0, f"qtip2 requires N % 64 == 0, K % {tile_k} == 0"
    g, kt = n // 64, k // tile_k
    nval = tile_k // 2
    dev = w.device
    row, col = _seq_coords2(dev, tile_k)
    rows = 64 * torch.arange(g, device=dev)[:, None, None, None] + row[None, None]
    cols = tile_k * torch.arange(kt, device=dev)[None, :, None, None] + col[None, None]
    seqs = w[rows.expand(g, kt, 128, nval), cols.expand(g, kt, 128, nval)]
    return seqs.reshape(g * kt * 128, nval).float()


def _viterbi2(x: torch.Tensor, values: torch.Tensor, c: torch.Tensor | None):
    """Exact Viterbi over the 2^16-state V=2 bitshift trellis for (B, 32, 2)
    targets (8-bit steps: s' = (s >> 8) | byte << 8, so predecessors of s'
    are the contiguous 256-block [(s' & 0xFF) << 8, +256)). c (B,) int64 or
    None: tail-biting constraint on the 8 shared boundary bits."""
    bs, S, _ = x.shape
    assert S in (16, 32)
    dev = x.device
    low8 = torch.arange(1 << 16, device=dev) & 0xFF

    def err(j):
        return (values[None, :, 0] - x[:, j, 0, None]) ** 2 + (
            values[None, :, 1] - x[:, j, 1, None]
        ) ** 2

    e0 = err(0)
    if c is None:
        cost = e0
    else:
        cost = torch.full((bs, 1 << 16), torch.inf, device=dev)
        idx0 = c[:, None] | (torch.arange(256, device=dev)[None, :] << 8)
        cost.scatter_(1, idx0, e0.gather(1, idx0))
    pre_args = []
    for j in range(1, S):
        pre_min, pre_arg = cost.view(bs, 256, 256).min(dim=-1)
        pre_args.append(pre_arg.to(torch.uint8))
        cost = err(j) + pre_min[:, low8]
    states = torch.zeros(bs, S, dtype=torch.int64, device=dev)
    if c is None:
        states[:, S - 1] = cost.argmin(dim=-1)
    else:
        idx_f = (c[:, None] << 8) | torch.arange(256, device=dev)[None, :]
        states[:, S - 1] = idx_f.gather(1, cost.gather(1, idx_f).argmin(dim=-1, keepdim=True))[:, 0]
    for j in range(S - 1, 0, -1):
        q = states[:, j] & 0xFF
        e = pre_args[j - 1].gather(1, q[:, None])[:, 0].to(torch.int64)
        states[:, j - 1] = (q << 8) | e
    return states


def quantize_qtip2_reference(
    w: torch.Tensor, batch_size: int = 8192, tile_k: int = 128, values: torch.Tensor | None = None
):
    """(N, K) float -> ((N/64, K/tile_k, 128, tile_k/4) uint8 stream blob,
    None).

    Exact 2^16-state V=2 Viterbi, tail-biting via the two-pass roll trick
    (pins the 8 shared boundary bits). Values are quantized at codebook
    scale — pre-scale w and carry the scale to the GEMM epilogue."""
    n, k = w.shape
    S = tile_k // 4  # trellis steps = stream bytes per sequence
    x = _gather_sequences2(w, tile_k).view(-1, S, 2)
    if values is None:
        values = qtip2_state_values(w.device)
    streams = []
    for i in range(math.ceil(x.shape[0] / batch_size)):
        xb = x[i * batch_size : (i + 1) * batch_size]
        s_roll = _viterbi2(xb.roll(S // 2, dims=1), values, None)
        c = s_roll[:, S // 2] & 0xFF
        states = _viterbi2(xb, values, c)
        assert ((states[:, 1:] & 0xFF) == (states[:, :-1] >> 8)).all()
        assert ((states[:, 0] & 0xFF) == (states[:, -1] >> 8)).all()
        streams.append((states & 0xFF).to(torch.uint8))
    blob = torch.cat(streams).view(n // 64, k // tile_k, 128, S)
    return blob, None


def dequant_qtip2_reference(blob: torch.Tensor, sf=None, values: torch.Tensor | None = None):
    """(N/64, K/tile_k, 128, tile_k/4) uint8 stream blob -> (N, K) f32
    (tile_k inferred from the stream length: 32 bytes -> 128, 16 -> 64)."""
    assert blob.dtype == torch.uint8 and blob.dim() == 4 and blob.shape[2] == 128
    S = blob.shape[3]
    assert S in (16, 32)
    tile_k = 4 * S
    g, kt = blob.shape[:2]
    dev = blob.device
    if values is None:
        values = qtip2_state_values(dev)
    byts = blob.to(torch.int64)
    j = torch.arange(S, device=dev)
    s = byts[..., j] | (byts[..., (j + 1) % S] << 8)  # (g, kt, 128, S) states
    vals = values[s.reshape(-1)].view(g, kt, 128, S, 2)
    vals = vals.view(g, kt, 128, 2 * S)  # value order: (state j, e) -> 2j + e
    out = torch.empty(g * 64, kt * tile_k, dtype=torch.float32, device=dev)
    row, col = _seq_coords2(dev, tile_k)
    rows = 64 * torch.arange(g, device=dev)[:, None, None, None] + row[None, None]
    cols = tile_k * torch.arange(kt, device=dev)[None, :, None, None] + col[None, None]
    out[rows.expand(g, kt, 128, 2 * S), cols.expand(g, kt, 128, 2 * S)] = vals
    return out


def dequant_qtip_reference(blob: torch.Tensor, sf=None) -> torch.Tensor:
    """(N/64, K/64, 128, 16) uint8 stream blob -> (N, K) f32."""
    assert blob.dtype == torch.uint8 and blob.dim() == 4 and blob.shape[2:] == (128, 16)
    g, kt = blob.shape[:2]
    dev = blob.device
    words = (blob.to(torch.int64).view(g, kt, 128, 4, 4) << (torch.arange(4, device=dev) * 8)).sum(
        -1
    )  # (g, kt, 128, 4) little-endian u32
    j = torch.arange(32, device=dev)
    wi, sh = j >> 3, 4 * (j & 7)
    lo = words[..., wi]
    hi = words[..., (wi + 1) % 4]
    win = ((lo >> sh) | (hi << (32 - sh))) & 0xFFFF  # sh=0: hi<<32 drops out
    vals = qtip_state_values(dev)[win.reshape(-1)].view(g, kt, 128, 32)
    out = torch.empty(g * 64, kt * 64, dtype=torch.float32, device=dev)
    row, col = _seq_coords(dev)
    rows = 64 * torch.arange(g, device=dev)[:, None, None, None] + row[None, None]
    cols = 64 * torch.arange(kt, device=dev)[None, :, None, None] + col[None, None]
    out[rows.expand(g, kt, 128, 32), cols.expand(g, kt, 128, 32)] = vals
    return out
