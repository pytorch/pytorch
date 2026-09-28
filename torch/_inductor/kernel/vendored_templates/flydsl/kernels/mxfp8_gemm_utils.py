# SPDX-License-Identifier: Apache-2.0
# Vendored, unmodified apart from this header, from the MXFP8 grouped GEMM
# package in ROCm/AMD-TorchTitan-Ops
# (`amd_titan/_ops/fwdgemm/_fp8_gemm_utils.py`).
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FP8-GEMM data-path helpers for the flydsl_8wave candidate. Self-contained:
# the only dependency is the `flydsl` pip wheel (flydsl.*).
#
# TRIMMED TO WHAT `mxfp8_grouped_gemm_gfx950` IMPORTS. Upstream also carries
# `preshuffle_b`, `ceildiv`, `divmod`, `StoreC` and `Mfma16x16x128`; this
# kernel imports none of them (it issues its own scaled MFMA through
# `_mfma_scale_agpr` and stores via `buffer_ops.buffer_store`), so they are
# dropped rather than vendored unused. `Mfma16x16x128` in particular built an
# `MFMA_Scale` atom and left its scale operands at the atom default, which is
# only correct while FlyDSL treats that default as an identity scale -- an
# assumption no live code here depends on.

import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect, llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec


def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


def swizzle_128(row, col):
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16)
                + (row % 16) * 16
                + (col // 64) * 1024
                + ((col % 64) // 16) * 256
                + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


class G2SLoader:
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

    def load_one(self, lds_dst, k_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))


class S2RLoader:
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)


def wait_barrier(count):
    """`s_waitcnt vmcnt(count) lgkmcnt(0)` then `s_barrier`.

    THE lgkmcnt(0) IS LOAD-BEARING. This barrier protects LDS buffers that other
    waves are about to overwrite, and vmcnt alone does not cover the ds_reads
    this wave still has in flight -- ds_read retires on lgkmcnt. The `s_barrier`
    lives inside an inline-asm string, so the backend cannot see a barrier here
    and will not insert the lgkmcnt(0) it normally places ahead of one.

    Without it the kernel is correct only by accident, on instruction distance:
    an s2r fragment is issued in one cluster and consumed in the NEXT, so it
    crosses this barrier outstanding, and the buffer it reads (`ac1`) is
    overwritten two clusters later by another wave that this same barrier just
    released. What covers the gap is the MFMAs in between -- N_ACCUMS of them.
    At N_ACCUMS 16 (256x256) and 8 (128x256, 256x128) the read always retires
    first. At N_ACCUMS 4 it does not, which is exactly the set that failed:
    (128,128) and BLOCK_R=64, 0.01-0.10% of elements wrong, varying run to run
    and only after repeated calls. Nothing else distinguished those two tiles --
    occupancy, the interleave plan and the swizzle round count were each ruled
    out by experiment first.
    """
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count}) lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
    )
