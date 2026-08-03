# SPDX-License-Identifier: BSD-3-Clause

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr


@dataclass(frozen=True)
class MXFP8GemmParams:
    """Compile-time identity for one specialized MXFP8 baseline kernel."""

    m: int
    n: int
    k: int
    out_dtype: str

    def __cache_signature__(self):
        return ("mxfp8_gfx950_v1", self.m, self.n, self.k, self.out_dtype)


MXFP8_BLOCK_K = 32
MXFP8_MFMA_K = 128
MXFP8_MFMA_M = 16
MXFP8_MFMA_N = 16
GFX950_WAVE_SIZE = 64


@functools.lru_cache(maxsize=64)
def make_mxfp8_scaled_mm_gfx950(*, m: int, n: int, k: int, out_dtype: str):
    """Build a correctness-first gfx950 MXFP8 scaled GEMM launcher.

    One Wave64 computes one 16x16 output tile. Inputs are contiguous E4M3
    matrices in [M, K] and [N, K] storage order. Scales are contiguous raw
    E8M0 bytes in [M, K / 32] and [N, K / 32] order.
    """
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if m % 32 != 0 or n % 32 != 0 or k % MXFP8_MFMA_K != 0:
        raise ValueError("MXFP8 baseline requires M%32 == N%32 == 0 and K%128 == 0")

    if out_dtype == "bfloat16":
        out_elem = fx.BFloat16
    elif out_dtype == "float16":
        out_elem = fx.Float16
    else:
        raise ValueError(f"unsupported MXFP8 output dtype: {out_dtype}")

    scale_k = k // MXFP8_BLOCK_K
    k_iters = k // MXFP8_MFMA_K
    grid_size = (m // MXFP8_MFMA_M) * (n // MXFP8_MFMA_N)

    @flyc.kernel(known_block_size=[GFX950_WAVE_SIZE, 1, 1])
    def kernel(
        a: fx.Tensor,
        b_nk: fx.Tensor,
        scale_a_u8: fx.Tensor,
        scale_b_u8: fx.Tensor,
        out: fx.Tensor,
    ):
        a_u8 = fx.recast_iter(fx.Uint8, fx.get_iter(a))
        b_u8 = fx.recast_iter(fx.Uint8, fx.get_iter(b_nk))
        sa_u8 = fx.recast_iter(fx.Uint8, fx.get_iter(scale_a_u8))
        sb_u8 = fx.recast_iter(fx.Uint8, fx.get_iter(scale_b_u8))
        out_ptr = fx.recast_iter(out_elem, fx.get_iter(out))

        lane = fx.Int32(fx.thread_idx.x)
        lane_16 = lane % fx.Int32(MXFP8_MFMA_N)
        k_group = lane // fx.Int32(MXFP8_MFMA_N)

        tiles_n = n // MXFP8_MFMA_N
        linear_tile = fx.Int32(fx.block_idx.x)
        tile_m = linear_tile // fx.Int32(tiles_n)
        tile_n = linear_tile % fx.Int32(tiles_n)
        m0 = tile_m * fx.Int32(MXFP8_MFMA_M)
        n0 = tile_n * fx.Int32(MXFP8_MFMA_N)
        a_row = m0 + lane_16
        b_row = n0 + lane_16

        mma_atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                MXFP8_MFMA_M,
                MXFP8_MFMA_N,
                MXFP8_MFMA_K,
                fx.Float8E4M3FN,
                fx.Float8E4M3FN,
                fx.Float32,
                opsel_a=0,
                opsel_b=0,
            )
        )
        c_frag = fx.make_rmem_tensor(4, fx.Float32)
        c_frag.store(fx.Vector.filled(4, 0.0, fx.Float32))
        vec16_u8_ty = fx.Vector.make_type(16, fx.Uint8)

        def load_operand_i32x8(base_u8, row, k_tile):
            row_base = fx.Int64(row) * fx.Int64(k)
            k_base = fx.Int64(k_tile * MXFP8_MFMA_K) + fx.Int64(k_group) * fx.Int64(16)
            lo = fx.ptr_load(
                base_u8 + row_base + k_base, result_type=vec16_u8_ty
            ).bitcast(fx.Int32)
            hi = fx.ptr_load(
                base_u8 + row_base + k_base + fx.Int64(64),
                result_type=vec16_u8_ty,
            ).bitcast(fx.Int32)
            packed = lo.shuffle(hi, list(range(8)))
            frag = fx.make_rmem_tensor(8, fx.Int32)
            frag.store(fx.Vector(packed))
            return frag

        def load_scale_word(scale_u8, row, k_tile):
            scale_offset = (
                fx.Int64(row) * fx.Int64(scale_k)
                + fx.Int64(k_tile * 4)
                + fx.Int64(k_group)
            )
            scale_byte = fx.ptr_load(scale_u8 + scale_offset)
            return scale_byte.to(fx.Int32) * fx.Int32(0x01010101)

        for k_tile in range_constexpr(k_iters):
            a_frag = load_operand_i32x8(a_u8, a_row, k_tile)
            b_frag = load_operand_i32x8(b_u8, b_row, k_tile)
            scale_a_word = load_scale_word(sa_u8, a_row, k_tile)
            scale_b_word = load_scale_word(sb_u8, b_row, k_tile)
            fx.gemm(
                mma_atom,
                c_frag,
                a_frag,
                b_frag,
                c_frag,
                scale_a=scale_a_word,
                scale_b=scale_b_word,
            )

        out4 = fx.Vector(c_frag.load().ir_value()).to(out_elem)
        col = n0 + lane_16
        row0 = m0 + k_group * fx.Int32(4)
        for i in range_constexpr(4):
            out_offset = fx.Int64(row0 + fx.Int32(i)) * fx.Int64(n) + fx.Int64(col)
            fx.ptr_store(out4[i], out_ptr + out_offset)

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b_nk: fx.Tensor,
        scale_a_u8: fx.Tensor,
        scale_b_u8: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(a, b_nk, scale_a_u8, scale_b_u8, out).launch(
            grid=(grid_size, 1, 1),
            block=(GFX950_WAVE_SIZE, 1, 1),
            stream=stream,
        )

    return launch
