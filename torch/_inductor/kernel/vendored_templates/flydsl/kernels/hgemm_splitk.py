# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
from abc import ABC, abstractmethod

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor, get_dtype_in_kernel

SPLIT_K_SEMAPHORE_MAX_LEN = 256


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


class WmmaHalfBase(ABC):
    @abstractmethod
    def __init__(self, dtype: str):
        pass

    @abstractmethod
    def __call__(self, a_frag, b_frag, c_frag):
        pass


class WmmaHalf_m16n16k16(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_A_FRAG_VALUES = 4
    WMMA_B_FRAG_VALUES = 4
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a_frag_vi16 = vector.bitcast(T.vec(self.WMMA_A_FRAG_VALUES, T.i16), a_frag)
            b_frag_vi16 = vector.bitcast(T.vec(self.WMMA_B_FRAG_VALUES, T.i16), b_frag)
            return rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [a_frag_vi16, b_frag_vi16, c_frag, 0, 0, 0]
            )
        return rocdl.mfma_f32_16x16x16f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
        )


class WmmaHalf_m16n16k32(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_A_FRAG_VALUES = 8
    WMMA_B_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            return rocdl.mfma_f32_16x16x32_bf16(
                T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
            )
        return rocdl.mfma_f32_16x16x32_f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
        )


class OnlineScheduler:
    def __init__(self, total_signals: int, init_count: int = 0):
        self.total_signals = total_signals
        self.current_signal_id = init_count
        self.remaining = init_count

    def release(self, count: int):
        count = min(count, self.total_signals - self.current_signal_id)
        self.current_signal_id += count
        self.remaining += count

    def consume(self, count: int):
        count = min(count, self.remaining)
        self.remaining -= count
        return count


@functools.lru_cache(maxsize=16384)
def compile_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    STAGES: int = 2,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    BLOCK_K_WARPS: int = 1,
    B_TO_LDS: bool = False,
    HAS_BIAS: bool = False,
):
    assert BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS <= 16
    assert TILE_M * TILE_N * TILE_K <= 256 * 256 * 64
    if (TILE_M == 256) and (TILE_N == 256):
        assert (TILE_K == 64) and (SPLIT_K == 1) and (STAGES == 2)
    assert STAGES >= 2
    N_BLOCKS = n // TILE_N
    assert (N_BLOCKS >= 1) and (n % TILE_N == 0)
    IS_SPLIT_K = SPLIT_K > 1
    IS_SLICE_K = BLOCK_K_WARPS > 1
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    GPU_ARCH = get_rocm_arch()
    if GPU_ARCH == "gfx942":
        WMMA_IMPL = WmmaHalf_m16n16k16(dtype)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
        ASYNC_COPY = True
    else:
        WMMA_IMPL = WmmaHalf_m16n16k32(dtype)
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
        ASYNC_COPY = True

    # Fixed parameters:
    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8

    # Propagated parameters:
    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    assert BLOCK_K_LOOPS >= STAGES
    WARP_GROUP_K = BLOCK_K_WARPS * WARP_ATOM_K
    WARP_K_STEPS = BLOCK_K // WARP_GROUP_K
    assert (BLOCK_K % WARP_GROUP_K == 0) and (WARP_K_STEPS >= 1)
    K_SLICE = BLOCK_K // BLOCK_K_WARPS
    assert K_SLICE % WARP_ATOM_K == 0
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE
    BLOCK_MN_WARPS = BLOCK_M_WARPS * BLOCK_N_WARPS
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert (n >= BLOCK_N) and (n % BLOCK_N == 0)
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    # LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1) and (LDG_REG_C_COUNT >= 1)
    assert BLOCK_MK_SIZE % BLOCK_VECS == 0
    assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # LDS parameters:
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_a_offset = allocator._align(allocator.ptr, 16)
    AS_BYTES = STAGES * BLOCK_M * BLOCK_K * DTYPE_BYTES
    allocator.ptr = smem_a_offset + AS_BYTES
    SMEM_USE = AS_BYTES
    if B_TO_LDS:
        smem_b_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = smem_b_offset + STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
        SMEM_USE += STAGES * BLOCK_N * BLOCK_K * DTYPE_BYTES
        assert ASYNC_COPY
    SMEM_USE_ = max(SMEM_USE, BLOCK_K_WARPS * BLOCK_M * BLOCK_N * DTYPE_BYTES)
    allocator.ptr += SMEM_USE_ - SMEM_USE
    assert SMEM_USE_ <= SMEM_CAPACITY_MAP[GPU_ARCH]
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_B_COUNT_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_WAIT_COUNT = LDG_REG_B_COUNT_AS + LDG_REG_A_COUNT_AS
    assert ((STAGES - 2) * LDG_WAIT_COUNT) < 63

    USE_8WAVE_PIPE = (
        ASYNC_COPY
        and B_TO_LDS
        and BLOCK_M == 256
        and BLOCK_N == 256
        and BLOCK_K == 64
        and LDG_REG_A_COUNT_AS == 4
        and LDG_REG_B_COUNT_AS == 4
        and MFMA_PER_WARP_K == 1
    )
    USE_8WAVE_PIPE = (
        USE_8WAVE_PIPE
        and BLOCK_M_WARPS == 2
        and BLOCK_N_WARPS == 4
        and BLOCK_K_WARPS == 1
    )

    KERNEL_NAME = f"hgemm_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}x{STAGES}_SPK{SPLIT_K}_W{BLOCK_M_WARPS}x{BLOCK_N_WARPS}x{BLOCK_K_WARPS}_BLDS{int(B_TO_LDS)}_TN"
    KERNEL_NAME += "_AS0" if not ASYNC_COPY else "_AS1"
    if HAS_BIAS:
        KERNEL_NAME += "_BIAS"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        if const_expr(HAS_BIAS):
            BIAS_ = GTensor(BIAS, dtype=dtype_, shape=(n,))
        base_ptr = allocator.get_base()
        smem_a_ptr = SmemPtr(
            base_ptr, smem_a_offset, dtype_, shape=(STAGES * BLOCK_M * BLOCK_K,)
        )
        as_ = STensor(smem_a_ptr, dtype_, shape=(STAGES, BLOCK_M, BLOCK_K))
        if const_expr(B_TO_LDS):
            smem_b_ptr = SmemPtr(
                base_ptr, smem_b_offset, dtype_, shape=(STAGES * BLOCK_N * BLOCK_K,)
            )
            bs_ = STensor(smem_b_ptr, dtype_, shape=(STAGES, BLOCK_N, BLOCK_K))
        smem_c_ptr = SmemPtr(
            base_ptr, smem_a_offset, dtype_, shape=(BLOCK_K_WARPS * BLOCK_M * BLOCK_N,)
        )
        cs_ = STensor(smem_c_ptr, dtype_, shape=(BLOCK_K_WARPS, BLOCK_M, BLOCK_N))
        if const_expr(IS_SPLIT_K):
            semaphore_ = GTensor(semaphore, dtype=T.i32, shape=(-1,))
            signal_ = GTensor(signal, dtype=T.i32, shape=(-1,))
            signal_idx = fx.Int32(fx.block_idx.x)

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        wid_mn = wid % BLOCK_MN_WARPS
        wid_k = wid // BLOCK_MN_WARPS
        w_tid = tid % WARP_SIZE

        def swizzle_for_cache_reuse(pid):
            # Do nothing currently
            return pid // N_BLOCKS, pid % N_BLOCKS

        block_m_idx, block_n_idx = swizzle_for_cache_reuse(fx.block_idx.x)
        ks_idx = fx.Index(fx.block_idx.y)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid_mn // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid_mn % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        warp_k_slice_base = wid_k * K_SLICE
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        def __barrier(vmcnt=0, use_s_barrier=True):
            if const_expr(use_s_barrier):
                asm = f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier"
            else:
                asm = f"s_waitcnt vmcnt({vmcnt})"
            llvm.InlineAsmOp(None, [], asm, "", has_side_effects=True)

        def get_llvm_ptr(
            ptr, offset, dtype_bytes, ptr_type=ir.Type.parse("!llvm.ptr<1>")
        ):
            base_ptr = arith.index_cast(T.i64, fx.ptrtoint(ptr))
            byte_offset = arith.index_cast(
                T.i64, fx.Index(offset) * fx.Index(dtype_bytes)
            )
            llvm_ptr = llvm.AddOp(
                base_ptr, byte_offset, llvm.IntegerOverflowFlags(0)
            ).result
            llvm_ptr = llvm.IntToPtrOp(ptr_type, llvm_ptr).result
            ptr_v = (
                llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr
            )
            return ptr_v

        def zero_c():
            # zero c if current block is the first block
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(0))
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = global_tid // LDG_C_X_THREADS
                    n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE
                    row_idx = m_offset + fx.Index(m_local_idx)
                    init_vec = zero_vec
                    if const_expr(HAS_BIAS):
                        init_vec = BIAS_.vec_load(
                            (n_offset + n_local_idx,), LDG_VEC_SIZE
                        )
                    cond_boundary = arith.cmpi(
                        arith.CmpIPredicate.ult, row_idx, fx.Index(m)
                    )
                    cond_boundary_if = scf.IfOp(
                        cond_boundary, results_=[], has_else=False
                    )
                    with ir.InsertionPoint(cond_boundary_if.then_block):
                        bytes_offset = C_.linear_offset(
                            (row_idx, n_offset + n_local_idx)
                        )
                        bytes_offset_i32 = arith.index_cast(T.i32, bytes_offset)
                        c_ptr = get_llvm_ptr(C, bytes_offset_i32, DTYPE_BYTES)
                        llvm.InlineAsmOp(
                            None,
                            [c_ptr, init_vec],
                            "global_store_dwordx4 $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        scf.YieldOp([])
                gpu.barrier()
                # trigger signal when zeroc is done by the first arrived block
                is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_cond_if.then_block):
                    signal_ptr = get_llvm_ptr(signal, signal_idx, 4)
                    llvm.InlineAsmOp(
                        None,
                        [signal_ptr, arith.constant(1, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1",
                        "v,v",
                        has_side_effects=True,
                    )
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([])

        def split_k_barrier():
            # spin-wait until signal triggered
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                init_cur = arith.constant(0, type=T.i32)
                w = scf.WhileOp([T.i32], [init_cur])
                before = ir.Block.create_at_start(w.before, [T.i32])
                after = ir.Block.create_at_start(w.after, [T.i32])
                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need_wait = arith.CmpIOp(
                        arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)
                    ).result
                    scf.ConditionOp(need_wait, [cur])
                with ir.InsertionPoint(after):
                    signal_ptr = get_llvm_ptr(signal, signal_idx, 4)
                    data = llvm.InlineAsmOp(
                        T.i32,
                        [signal_ptr],
                        "global_load_dword $0, $1, off sc1",
                        "=v,v",
                        has_side_effects=True,
                    ).result
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([data])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()
            # clean semaphore and signal if this is the last block within split-k group
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                semaphore_ptr = get_llvm_ptr(semaphore, signal_idx, 4)
                arrive_idx = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    semaphore_ptr,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                cond_ksl = arith.cmpi(
                    arith.CmpIPredicate.eq, fx.Index(arrive_idx), fx.Index(SPLIT_K - 1)
                )
                cond_ksl_if = scf.IfOp(cond_ksl, results_=[], has_else=False)
                with ir.InsertionPoint(cond_ksl_if.then_block):
                    semaphore_[signal_idx] = arith.constant(0, type=T.i32)
                    signal_[signal_idx] = arith.constant(0, type=T.i32)
                    scf.YieldOp([])
                scf.YieldOp([])
            gpu.barrier()

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vec = A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs

        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                as_.vec_store(
                    (fx.Index(lds_stage), m_local_idx, col_in_bytes // DTYPE_BYTES),
                    vecs[i],
                    LDG_VEC_SIZE,
                )

        def get_dma_copy_warp_offset():
            warp_offset = rocdl.readfirstlane(
                T.i64,
                arith.index_cast(
                    T.i64,
                    fx.Index(wid) * arith.constant(WARP_SIZE * DMA_BYTES, index=True),
                ),
            )
            return warp_offset

        def buffer_load_lds_inline(rsrc, lds_ptr, global_offset):
            if const_expr(DMA_BYTES == 16):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds"
            elif const_expr(DMA_BYTES == 8):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx2 $1, $2, 0 offen sc0 lds"
            elif const_expr(DMA_BYTES == 4):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, 0 offen sc0 lds"
            else:
                raise NotImplementedError(f"DMA_BYTES={DMA_BYTES} not supported")
            llvm.InlineAsmOp(
                None,
                [lds_ptr, global_offset, rsrc],
                asm,
                "s,v,s",
                has_side_effects=True,
            )

        def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            m_local_idx = global_tid // LDG_A_X_THREADS_AS
            k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
            row_idx = m_offset + fx.Index(m_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                row_idx,
                fx.Index(0),
            )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = A_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(lds_ptr is None):
                lds_offset = (
                    as_.linear_offset((fx.Index(write_stage), 0, 0)) * DTYPE_BYTES
                )
                lds_base = (
                    memref.extract_aligned_pointer_as_index(as_.memptr) + lds_offset
                )
                lds_ptr_base = buffer_ops.create_llvm_ptr(
                    arith.index_cast(T.i64, lds_base), address_space=3
                )
                lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
                )
            buffer_load_lds_inline(A_.rsrc, lds_ptr, global_offset)
            return lds_ptr

        def ldg_sts_a_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                lds_ptr = ldg_sts_a_async_one(
                    i, k_offset, lds_stage, lds_ptr if i > 0 else None
                )

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_B_X_THREADS_AS
            k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            row_idx = n_offset + fx.Index(n_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(n)),
                row_idx,
                fx.Index(0),
            )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = B_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(lds_ptr is None):
                lds_offset = (
                    bs_.linear_offset((fx.Index(write_stage), 0, 0)) * DTYPE_BYTES
                )
                lds_base = (
                    memref.extract_aligned_pointer_as_index(bs_.memptr) + lds_offset
                )
                lds_ptr_base = buffer_ops.create_llvm_ptr(
                    arith.index_cast(T.i64, lds_base), address_space=3
                )
                lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
                )
            buffer_load_lds_inline(B_.rsrc, lds_ptr, global_offset)
            return lds_ptr

        def ldg_sts_b_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(LDG_REG_B_COUNT_AS):
                lds_ptr = ldg_sts_b_async_one(
                    i, k_offset, lds_stage, lds_ptr if i > 0 else None
                )

        def ldg_matrix_b(k_offset):
            vecs = []
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                    n_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                    k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    vec = B_.vec_load(
                        (n_idx, k_idx), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
                    )
                    vecs.append(vec)
            return vecs

        def ldmatrix_compute_tile_streaming(lds_stage, c_frags, initial_b_frags=None):
            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                if const_expr(initial_b_frags is None):
                    b_frags = [0] * WARP_N_STEPS
                    for ii in range_constexpr(WARP_N_STEPS):
                        warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                        row = warp_atom_n_idx + ldmatrix_b_n_idx
                        col_in_bytes = (
                            warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        ) * DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        vec = bs_.vec_load(
                            (s, row, col_in_bytes // DTYPE_BYTES),
                            WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                        b_frags[ii] = vec
                else:
                    b_frags = [
                        initial_b_frags[i]
                        for i in range_constexpr(
                            kk * WARP_N_STEPS, (kk + 1) * WARP_N_STEPS
                        )
                    ]
                a_frags = [0] * WARP_M_STEPS
                for ii in range_constexpr(WARP_M_STEPS):
                    warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_a_k_vec_idx
                    ) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_.vec_load(
                        (s, row, col_in_bytes // DTYPE_BYTES),
                        WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                    )
                    a_frags[ii] = vec
                rocdl.sched_barrier(0)
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[jj]
                        if const_expr(MFMA_PER_WARP_K == 2):
                            # split a
                            a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                            a0_i64 = vector.extract(
                                a_i64x2, static_position=[0], dynamic_position=[]
                            )
                            a1_i64 = vector.extract(
                                a_i64x2, static_position=[1], dynamic_position=[]
                            )
                            a_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64])
                            )
                            a_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64])
                            )
                            # split b
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                            b0_i64 = vector.extract(
                                b_i64x2, static_position=[0], dynamic_position=[]
                            )
                            b1_i64 = vector.extract(
                                b_i64x2, static_position=[1], dynamic_position=[]
                            )
                            b_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64])
                            )
                            b_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64])
                            )
                            # wmma
                            c_idx = ii * WARP_N_STEPS + jj
                            acc_in = c_frags_new[c_idx]
                            acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                            c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                        elif const_expr(MFMA_PER_WARP_K == 1):
                            c_idx = ii * WARP_N_STEPS + jj
                            c_frags_new[c_idx] = WMMA_IMPL(
                                a_frag, b_frag, c_frags_new[c_idx]
                            )
                        else:
                            raise NotImplementedError(
                                f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported"
                            )
            return c_frags_new

        def async_copy_ldmatrix_compute_tile_streaming(
            lds_stage,
            c_frags,
            k_offset_next_tile,
            write_stage,
        ):
            assert LDG_REG_A_COUNT_AS == 4
            assert LDG_REG_B_COUNT_AS == 4
            assert MFMA_PER_WARP_K == 1
            assert WARP_M_STEPS % 2 == 0
            assert WARP_N_STEPS % 2 == 0

            M_HALF_STEPS = WARP_M_STEPS // 2
            N_HALF_STEPS = WARP_N_STEPS // 2

            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            lds_ptr_a = None
            lds_ptr_b = None

            def load_b_frag(n_step, warp_atom_k_idx):
                warp_atom_n_idx = warp_n_idx + n_step * WARP_ATOM_N
                row = warp_atom_n_idx + ldmatrix_b_n_idx
                col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                return bs_.vec_load(
                    (s, row, col_in_bytes // DTYPE_BYTES),
                    WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                )

            def load_a_frag(m_step, warp_atom_k_idx):
                warp_atom_m_idx = warp_m_idx + m_step * WARP_ATOM_M
                row = warp_atom_m_idx + ldmatrix_a_m_idx
                col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                return as_.vec_load(
                    (s, row, col_in_bytes // DTYPE_BYTES),
                    WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                )

            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = kk * WARP_ATOM_K
                b0_frags = [0] * N_HALF_STEPS  # 2
                b1_frags = [0] * N_HALF_STEPS
                a0_frags = [0] * M_HALF_STEPS  # 4
                a1_frags = [0] * M_HALF_STEPS
                if const_expr(kk == 0):
                    lds_ptr_b = ldg_sts_b_async_one(
                        0, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                    lds_ptr_b = ldg_sts_b_async_one(
                        1, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                for ni in range_constexpr(N_HALF_STEPS):
                    b0_frags[ni] = load_b_frag(ni, warp_atom_k_idx)
                for mi in range_constexpr(M_HALF_STEPS):
                    a0_frags[mi] = load_a_frag(mi, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = mi * WARP_N_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a0_frags[mi],
                            b0_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                if const_expr(kk == 0):
                    lds_ptr_a = ldg_sts_a_async_one(
                        0, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    lds_ptr_a = ldg_sts_a_async_one(
                        1, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                for ni in range_constexpr(N_HALF_STEPS):
                    b1_frags[ni] = load_b_frag(N_HALF_STEPS + ni, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = mi * WARP_N_STEPS + N_HALF_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a0_frags[mi],
                            b1_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                    lds_ptr_b = ldg_sts_b_async_one(
                        2, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                    lds_ptr_b = ldg_sts_b_async_one(
                        3, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                for mi in range_constexpr(M_HALF_STEPS):
                    a1_frags[mi] = load_a_frag(M_HALF_STEPS + mi, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = (M_HALF_STEPS + mi) * WARP_N_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a1_frags[mi],
                            b0_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                    lds_ptr_a = ldg_sts_a_async_one(
                        2, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    lds_ptr_a = ldg_sts_a_async_one(
                        3, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = (M_HALF_STEPS + mi) * WARP_N_STEPS + N_HALF_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a1_frags[mi],
                            b1_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
            return c_frags_new

        warp_offset = get_dma_copy_warp_offset()

        if const_expr(IS_SPLIT_K):
            zero_c()

        if const_expr(B_TO_LDS):

            for s in range_constexpr(STAGES - 1):
                ldg_sts_b_async(ks_begin + s * BLOCK_K, s)
                ldg_sts_a_async(ks_begin + s * BLOCK_K, s)
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                # ================ Ordered ================
                if const_expr(USE_8WAVE_PIPE):
                    for ki in range_constexpr(WARP_K_STEPS):
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_dsrd(4)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(4)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_mfma(8)
                else:
                    for i in range_constexpr(LDG_REG_B_COUNT_AS):
                        rocdl.sched_vmem(1)  # ldg_sts_b_async next
                    for i in range_constexpr(LDG_REG_A_COUNT_AS):
                        rocdl.sched_vmem(1)  # ldg_sts_a_async next
                    for ki in range_constexpr(WARP_K_STEPS):
                        for i in range_constexpr(WARP_N_STEPS):
                            rocdl.sched_dsrd(1)  # lds_matrix_b current
                        for i in range_constexpr(WARP_M_STEPS):
                            rocdl.sched_dsrd(1)  # lds_matrix_a current
                        for i in range_constexpr(WARP_M_STEPS):
                            rocdl.sched_mfma(WARP_N_STEPS)
                # ================ Reordered ================
                rocdl.sched_barrier(0)

            init_state = [ks_begin, arith.constant(0, index=True)] + c_frags
            for bki, state in range(
                0, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state
            ):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                c_frags = state[2:]
                next_stage = (current_stage + 1) % STAGES
                write_stage = (current_stage + STAGES - 1) % STAGES
                __barrier((STAGES - 2) * LDG_WAIT_COUNT)
                if const_expr(USE_8WAVE_PIPE):
                    c_frags_new = async_copy_ldmatrix_compute_tile_streaming(
                        current_stage,
                        c_frags,
                        k_offset + (STAGES - 1) * BLOCK_K,
                        write_stage,
                    )
                else:
                    ldg_sts_b_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
                    ldg_sts_a_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
                    c_frags_new = ldmatrix_compute_tile_streaming(
                        current_stage, c_frags
                    )
                k_offset_next = k_offset + fx.Int32(BLOCK_K)
                hot_loop_scheduler()
                results = yield [k_offset_next, next_stage] + c_frags_new
            current_stage = fx.Index(results[1])
            c_frags = results[2:]
            for s in range_constexpr(0, STAGES - 1):
                __barrier((STAGES - 2 - s) * LDG_WAIT_COUNT)
                c_frags = ldmatrix_compute_tile_streaming(current_stage, c_frags)
                current_stage = (current_stage + 1) % STAGES

        else:

            assert STAGES == 2
            sts_a(ldg_a(ks_begin), 0)
            b_frags_next = ldg_matrix_b(ks_begin)
            rocdl.sched_barrier(0)
            __barrier()

            def hot_loop_scheduler():
                LDG_REG_A_COUNT_ = (
                    LDG_REG_A_COUNT_AS if const_expr(ASYNC_COPY) else LDG_REG_A_COUNT
                )
                LDG_TOTAL = LDG_REG_A_COUNT_ + WARP_K_STEPS * WARP_N_STEPS
                # ================ Ordered ================
                for i in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(1)
                for ki in range_constexpr(WARP_K_STEPS):
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_dsrd(1)
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_mfma(WARP_N_STEPS)
                # ================ Reordered ================
                rocdl.sched_barrier(0)

            init_state = (
                [ks_begin, arith.constant(0, index=True)] + c_frags + b_frags_next
            )
            for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN :]
                if const_expr(ASYNC_COPY):
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                else:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
                c_frags_new = ldmatrix_compute_tile_streaming(
                    current_stage, c_frags, b_frags
                )
                if const_expr(not ASYNC_COPY):
                    sts_a(a_regs_next, next_stage)
                k_offset = k_offset + fx.Int32(BLOCK_K)
                hot_loop_scheduler()
                __barrier()
                results = yield [k_offset, next_stage] + c_frags_new + b_frags_next
            current_stage = fx.Index(results[1])
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN :]
            c_frags = ldmatrix_compute_tile_streaming(current_stage, c_frags, b_frags)

        # write to lds
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N
        gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                    lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                    val = vector.extract(
                        c_frags[ii * WARP_N_STEPS + jj],
                        static_position=[kk],
                        dynamic_position=[],
                    )
                    val = val.truncf(dtype_)
                    if const_expr(IS_SLICE_K):
                        cs_[wid_k, lds_m_idx, lds_n_idx] = val
                    else:
                        cs_[0, lds_m_idx, lds_n_idx] = val

        # write back to global
        if const_expr(IS_SPLIT_K):
            split_k_barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                n_global_idx = n_offset + n_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    pk_val = cs_.vec_load((0, m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        pk_val += cs_.vec_load(
                            (ksi, m_local_idx, n_local_idx), LDG_VEC_SIZE
                        )
                    linear_offset_c = C_.linear_offset((m_global_idx, n_global_idx))
                    # split to vec2s
                    vec2_ty = T.vec(2, dtype_)
                    for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                        e0 = vector.extract(
                            pk_val, static_position=[vec_idx * 2], dynamic_position=[]
                        )
                        e1 = vector.extract(
                            pk_val,
                            static_position=[vec_idx * 2 + 1],
                            dynamic_position=[],
                        )
                        pair = vector.from_elements(vec2_ty, [e0, e1])
                        pair_v = (
                            pair._value if const_expr(hasattr(pair, "_value")) else pair
                        )
                        pair_ptr_v = get_llvm_ptr(
                            C, fx.Int32(linear_offset_c + vec_idx * 2), DTYPE_BYTES
                        )
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            pair_ptr_v,
                            pair_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )
                    scf.YieldOp([])
        else:
            gpu.barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    vec = cs_.vec_load((0, m_local_idx, n_local_idx), LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        vec += cs_.vec_load(
                            (ksi, m_local_idx, n_local_idx), LDG_VEC_SIZE
                        )
                    if const_expr(HAS_BIAS):
                        bias_vec = BIAS_.vec_load(
                            (n_offset + n_local_idx,), LDG_VEC_SIZE
                        )
                        vec = vec + bias_vec
                    C_.vec_store(
                        (m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE
                    )
                    scf.YieldOp([])
        return

    @flyc.jit
    def launch_hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bm = (m + BLOCK_M - 1) // BLOCK_M
        hgemm_kernel._func.__name__ = KERNEL_NAME
        value_attrs = (
            {
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            }
            if USE_8WAVE_PIPE
            else None
        )
        hgemm_kernel(C, A, B, BIAS, m, semaphore, signal).launch(
            grid=(bm * N_BLOCKS, SPLIT_K, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
            value_attrs=value_attrs,
        )

    return launch_hgemm_kernel
