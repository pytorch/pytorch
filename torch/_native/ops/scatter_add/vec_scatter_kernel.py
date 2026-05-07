"""Vectorized scatter_add with per-element atomics (pre-sm_90 fallback).

Same input restriction as the TMA path (expanded-1D index, dim=0, 2D
contiguous tensors), but works everywhere scalar atomicAdd does. Matches
the fallback path in https://github.com/pytorch/pytorch/pull/182675.

Algorithm: warp-per-source-row. Each warp reads ``idx[entry]`` once, then
the 32 lanes chunk through N with vectorized gather (4 fp32 or 8 halves
per lane per step) and issue atomics into ``out[ind, :]``:

- fp32: per-element ``atomicAdd``.
- fp16/bf16: paired x2 atomics
  (``red.global.add.noftz.{f16x2,bf16x2}``), one instruction per pair.
  Matches the instruction count aten gets from ``atomicAdd(__half2*, ...)``.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector as mlir_vector
from cutlass.cutlass_dsl import dsl_user_op, T

import torch
from torch._vendor.quack.cache_utils import jit_cache

from ._ptx import make_packed_half_atomic_add


_WARPS_PER_BLOCK = 8
_THREADS_PER_BLOCK = 32 * _WARPS_PER_BLOCK
_VEC_BYTES = 16  # 16-byte vector gather = LDG.128

_TORCH_TO_CUTE = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
}


def _pack_half2_as_i32(val_a, val_b, elem_mlir_type, *, loc=None, ip=None):
    """Build a <2 x elem> vector from two scalar halves and bitcast to i32.
    Matches the layout PTX red.global.add.noftz.{f16x2,bf16x2} expects."""
    vec_type = ir.VectorType.get([2], elem_mlir_type, loc=loc)
    vec = mlir_vector.from_elements(
        vec_type,
        [val_a.ir_value(loc=loc, ip=ip), val_b.ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    return llvm.bitcast(T.i32(), vec, loc=loc, ip=ip)


_packed_atomic_add_f16x2 = make_packed_half_atomic_add("f16x2")
_packed_atomic_add_bf16x2 = make_packed_half_atomic_add("bf16x2")


@dsl_user_op
def _atomic_add_f16x2(ptr, val_a: Float16, val_b: Float16, *, loc=None, ip=None):
    """Packed x2 f16 atomic add. Equivalent to atomicAdd(__half2*, ...)."""
    gmem_ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    packed = _pack_half2_as_i32(val_a, val_b, Float16.mlir_type, loc=loc, ip=ip)
    _packed_atomic_add_f16x2(gmem_ptr_i64, packed, loc=loc, ip=ip)


@dsl_user_op
def _atomic_add_bf16x2(ptr, val_a: BFloat16, val_b: BFloat16, *, loc=None, ip=None):
    """Packed x2 bf16 atomic add (sm_90+)."""
    gmem_ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    packed = _pack_half2_as_i32(val_a, val_b, BFloat16.mlir_type, loc=loc, ip=ip)
    _packed_atomic_add_bf16x2(gmem_ptr_i64, packed, loc=loc, ip=ip)


def _make_kernel(dtype, elem_bytes: int, vec_elems: int, scale: bool):
    """Build a dtype-specialized vectorized scatter-add kernel.

    Each warp handles one source row. 32 lanes chunk through N, each lane
    loading vec_elems consecutive elements and atomic-adding them into
    out[ind, :].

    fp16/bf16 use paired x2 atomics (``red.global.add.noftz.{f16x2,bf16x2}``)
    to match the instruction count aten gets with ``atomicAdd(__half2*, ...)``.
    fp32 uses scalar ``atomicAdd``.

    When ``scale`` is True each src value is multiplied by the runtime
    ``alpha`` argument before the atomic. For alpha == 1 the caller should
    pick the non-scaling variant to skip the multiply entirely.
    """
    is_half = dtype is Float16 or dtype is BFloat16

    @cute.kernel
    def _kernel(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        num_entries: Int32,
        D: Int32,
        alpha: cute.Numeric,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        warp_in_block = tidx // Int32(32)
        lane = tidx - warp_in_block * Int32(32)

        # Alpha comes across the ABI as fp32 (tvm_ffi doesn't support
        # fp16/bf16 scalar args). Cast once to the src dtype so the
        # per-element multiply is in-dtype.
        if const_expr(scale and is_half):
            alpha_t = dtype(alpha)
        else:
            alpha_t = alpha

        entries_per_block = Int32(_WARPS_PER_BLOCK)
        base = bidx * entries_per_block
        while base < num_entries:
            entry_id = base + warp_in_block
            if entry_id < num_entries:
                idx_t = cute.make_tensor(
                    mIndex.iterator + Int64(entry_id) * Int64(mIndex.stride[0]),
                    cute.make_layout(1),
                )
                r = Int64(idx_t[0])

                lane_offset = lane * Int32(vec_elems)
                stride_elems = Int32(32 * vec_elems)

                off = lane_offset
                while off < D:
                    src_off = Int64(entry_id) * Int64(D) + Int64(off)
                    dst_off = r * Int64(D) + Int64(off)

                    if const_expr(is_half):
                        # Paired atomics: one x2 atomic per 2 elements.
                        # vec_elems is even for halves (8 for f16/bf16 at
                        # 16 bytes), so vec_elems // 2 is exact.
                        for p in cutlass.range_constexpr(vec_elems // 2):
                            v0 = 2 * p
                            v1 = 2 * p + 1
                            src_a = cute.make_tensor(
                                mSrc.iterator + (src_off + Int64(v0)),
                                cute.make_layout(1),
                            )
                            src_b = cute.make_tensor(
                                mSrc.iterator + (src_off + Int64(v1)),
                                cute.make_layout(1),
                            )
                            val_a = src_a[0]
                            val_b = src_b[0]
                            if const_expr(scale):
                                val_a = val_a * alpha_t
                                val_b = val_b * alpha_t
                            dst_ptr = mOut.iterator + (dst_off + Int64(v0))
                            if const_expr(dtype is Float16):
                                _atomic_add_f16x2(dst_ptr, val_a, val_b)
                            else:
                                _atomic_add_bf16x2(dst_ptr, val_a, val_b)
                    else:
                        for v in cutlass.range_constexpr(vec_elems):
                            src_t = cute.make_tensor(
                                mSrc.iterator + (src_off + Int64(v)),
                                cute.make_layout(1),
                            )
                            val = src_t[0]
                            if const_expr(scale):
                                val = val * alpha_t
                            dst_ptr_v = mOut.iterator + (dst_off + Int64(v))
                            cute.arch.atomic_add(dst_ptr_v, val)

                    off = off + stride_elems

            base = base + gdim * entries_per_block

    @cute.jit
    def _launch(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_entries: Int32,
        D: Int32,
        grid_x: Int32,
        alpha: cute.Numeric,
    ):
        _kernel(mSrc, mIndex, mOut, num_entries, D, alpha).launch(
            grid=[grid_x, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch


def _alpha_dtype_for(torch_dtype: torch.dtype):
    """Alpha is passed as fp32 across the ABI; the tvm_ffi runtime doesn't
    support fp16/bf16 scalars. For half dtypes the kernel casts alpha to
    the src dtype before multiplying."""
    return Float32


@jit_cache
def _compile_vec_scatter(torch_dtype: torch.dtype, N: int, scale: bool):
    dtype = _TORCH_TO_CUTE[torch_dtype]
    elem_bytes = dtype.width // 8
    vec_elems = _VEC_BYTES // elem_bytes
    launcher = _make_kernel(dtype, elem_bytes, vec_elems, scale)

    mSrc_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )
    mIndex_fake = cute.runtime.make_fake_tensor(
        Int64, (cute.sym_int(),), stride=(cute.sym_int64(),)
    )
    mOut_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), N), stride=(cute.sym_int64(), 1)
    )
    alpha_dtype = _alpha_dtype_for(torch_dtype)
    return cute.compile(
        launcher,
        mSrc_fake,
        mIndex_fake,
        mOut_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        Int32(0),
        alpha_dtype(0.0),
        options="--enable-tvm-ffi",
    )


def vec_elems_for(dtype: torch.dtype) -> int:
    """Elements per lane per step (16-byte vector gather)."""
    return _VEC_BYTES // torch.tensor([], dtype=dtype).element_size()


def vec_scatter_add_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
    alpha: float = 1.0,
) -> None:
    """In-place: ``out[index_1d[i], :] += alpha * src[i, :]`` for every i.

    Requires N divisible by ``vec_elems_for(src.dtype)`` (the cond enforces
    this). When N < 32 * vec_elems the loop has fewer active lanes per
    warp; lanes whose starting offset is >= N simply skip.

    When ``alpha == 1.0`` a non-scaling kernel variant is used (no runtime
    multiply in the hot loop).
    """
    M, N = src.shape
    scale = alpha != 1.0
    compiled = _compile_vec_scatter(src.dtype, N, scale)
    sm = torch.cuda.get_device_properties(out.device).multi_processor_count
    grid = min((M + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK, sm * 8)
    alpha_dtype = _alpha_dtype_for(src.dtype)
    compiled(src, index_1d, out, M, N, grid, alpha_dtype(float(alpha)))
