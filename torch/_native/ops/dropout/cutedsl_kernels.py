"""CuTeDSL fused dropout, bit-exact to aten's fused_dropout_kernel_vec (VEC=4).

Philox4x32-10 is implemented directly (10 rounds, curand counter layout:
counter.xy = offset/4 + iteration, counter.zw = subsequence = global thread
id, key = seed). Each thread consumes one uniform4 per grid-stride iteration
covering 4 contiguous elements, exactly like the aten kernel, so both the
values and the number of philox draws (hence the generator offset
advancement) match aten.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from ._common import _BLOCK, keep_prob_and_scale, launch_plan, philox_args


_PHILOX_10A = 0x9E3779B9
_PHILOX_10B = 0xBB67AE85
_PHILOX_SA = 0xD2511F53
_PHILOX_SB = 0xCD9E8D57

# Exact float32 values of curand's CURAND_2POW32_INV and its half.
_INV32 = 2.0**-32
_INV33 = 2.0**-33


@cute.jit
def _mulhilo(a: Uint32, b: Uint32) -> tuple[Uint32, Uint32]:
    prod = Uint64(a) * Uint64(b)
    return Uint32(prod & 0xFFFFFFFF), Uint32(prod >> 32)


@cute.jit
def _philox4(
    c0: Uint32, c1: Uint32, c2: Uint32, c3: Uint32, k0: Uint32, k1: Uint32
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    for _ in cutlass.range_constexpr(9):
        lo0, hi0 = _mulhilo(Uint32(_PHILOX_SA), c0)
        lo1, hi1 = _mulhilo(Uint32(_PHILOX_SB), c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + Uint32(_PHILOX_10A)
        k1 = k1 + Uint32(_PHILOX_10B)
    lo0, hi0 = _mulhilo(Uint32(_PHILOX_SA), c0)
    lo1, hi1 = _mulhilo(Uint32(_PHILOX_SB), c2)
    return hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0


@cute.jit
def _uniform(x: Uint32) -> Float32:
    return Float32(x) * Float32(_INV32) + Float32(_INV33)


class _FusedDropout:
    def __init__(self, num_iters: int):
        self.num_iters = num_iters

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mMask: cute.Tensor,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        intra: Int64,
        p_keep: Float32,
        scale: Float32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mOut, mMask, mSeed, mOff, intra, p_keep, scale).launch(
            grid=[grid_x, 1, 1], block=[_BLOCK, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mMask: cute.Tensor,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        intra: Int64,
        p_keep: Float32,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        n = mX.shape[0]
        tid = Int64(bidx) * _BLOCK + Int64(tidx)
        seed = Uint64(mSeed[0])
        off4 = (Int64(mOff[0]) + intra) >> 2
        k0 = Uint32(seed & 0xFFFFFFFF)
        k1 = Uint32(seed >> 32)
        c2 = Uint32(Uint64(tid) & 0xFFFFFFFF)
        c3 = Uint32(Uint64(tid) >> 32)
        stride = Int64(gdim) * _BLOCK * 4

        for it in cutlass.range_constexpr(self.num_iters):
            ctr = Uint64(off4 + it)
            u0, u1, u2, u3 = _philox4(
                Uint32(ctr & 0xFFFFFFFF), Uint32(ctr >> 32), c2, c3, k0, k1
            )
            base = tid * 4 + it * stride
            if base < n:
                r0 = _uniform(u0) < p_keep
                r1 = _uniform(u1) < p_keep
                r2 = _uniform(u2) < p_keep
                r3 = _uniform(u3) < p_keep
                f0 = Float32(1.0) if r0 else Float32(0.0)
                f1 = Float32(1.0) if r1 else Float32(0.0)
                f2 = Float32(1.0) if r2 else Float32(0.0)
                f3 = Float32(1.0) if r3 else Float32(0.0)
                mOut[base + 0] = mX[base + 0] * f0 * scale
                mOut[base + 1] = mX[base + 1] * f1 * scale
                mOut[base + 2] = mX[base + 2] * f2 * scale
                mOut[base + 3] = mX[base + 3] * f3 * scale
                mMask[base + 0] = Boolean(r0)
                mMask[base + 1] = Boolean(r1)
                mMask[base + 2] = Boolean(r2)
                mMask[base + 3] = Boolean(r3)


def _fake_flat(dtype, divisibility=1):
    return cute.runtime.make_fake_tensor(
        dtype,
        (cute.sym_int(divisibility=divisibility),),
        stride=(1,),
        assumed_align=max(divisibility * dtype.width // 8, 1),
    )


@instrumented_cutedsl_cache(
    "aten::native_dropout", key_fn=lambda num_iters: f"fused_dropout iters={num_iters}"
)
def _compile_dropout(num_iters: int):
    x_fake = _fake_flat(Float32, 4)
    out_fake = _fake_flat(Float32, 4)
    mask_fake = _fake_flat(Boolean, 4)
    state_fake = cute.runtime.make_fake_tensor(
        Int64, (1,), stride=(1,), assumed_align=8
    )
    state_fake2 = cute.runtime.make_fake_tensor(
        Int64, (1,), stride=(1,), assumed_align=8
    )
    return cute.compile(
        _FusedDropout(num_iters),
        x_fake,
        out_fake,
        mask_fake,
        state_fake,
        state_fake2,
        Int64(0),
        Float32(0.5),
        Float32(2.0),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def dropout_fwd(x: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.numel()
    grid, counter_offset, num_iters = launch_plan(n, x.device.index or 0)
    p_keep, scale = keep_prob_and_scale(p)
    seed_t, offset_t, intra = philox_args(x, counter_offset)
    out = torch.empty_like(x)
    mask = torch.empty_like(x, dtype=torch.bool)
    x_flat, out_flat, mask_flat = x.view(-1), out.view(-1), mask.view(-1)
    _compile_dropout(num_iters)(
        x_flat, out_flat, mask_flat, seed_t, offset_t, intra, p_keep, scale, grid
    )
    return out, mask
