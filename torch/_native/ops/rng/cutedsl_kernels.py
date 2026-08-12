"""CuTeDSL distribution kernels, bit-exact to aten's distribution_nullary_kernel.

Philox4x32-10 is implemented directly (10 rounds), with curand's counter layout:
counter.zw = subsequence = the thread's flat index, counter.xy = offset/4 + iteration,
key = seed. Each grid-stride iteration consumes one curand4 draw and writes `unroll`
STRIDED elements (idx + it*stride + ii*blockDim*gridDim), matching aten's loop exactly --
so both the values and the number of philox draws (hence the generator's offset
advancement) are identical to aten.

Mirroring PhiloxCudaState, each kernel is compiled in two variants: eager (HostState:
seed/offset arrive as Int64 scalars) and capture (DevState: seed/offset are loaded from the
generator's extragraph tensors at run time, so replays see what replay_prologue wrote).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from .._cutedsl import launch as _L
from ._common import _BLOCK, choose_iter_unroll


_PHILOX_10A = 0x9E3779B9
_PHILOX_10B = 0xBB67AE85
_PHILOX_SA = 0xD2511F53
_PHILOX_SB = 0xCD9E8D57

# curand's CURAND_2POW32_INV and its half: uniform = u*2^-32 + 2^-33 places the value in
# (0, 1) with the same rounding curand_uniform4 produces.
_INV32 = 2.0**-32
_INV33 = 2.0**-33


# curand's CURAND_2POW32_INV_2PI == float32(2^-32) * float32(2*pi), and its half. The
# box-muller angle is y*S + S/2 with S folded like this; scaling a uniform by 2*pi instead
# rounds differently and loses bit-exactness (curand_globals.h / curand_normal.h).
_INV_2PI = 1.462918119976564e-09
_INV_2PI_HALF = 7.31459059988282e-10


@cute.jit
def _mulhilo(a: Uint32, b: Uint32) -> tuple[Uint32, Uint32]:
    prod = Uint64(a) * Uint64(b)
    return Uint32(prod & 0xFFFFFFFF), Uint32(prod >> 32)


@cute.jit
def _philox4(
    c0: Uint32, c1: Uint32, c2: Uint32, c3: Uint32, k0: Uint32, k1: Uint32
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    # 10 rounds: 9 with key bumps, then a final round (curand_Philox4x32_10).
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


class _Distribution:
    # kind: "uniform" -> a*u + b (uniform_ / rand); "normal" -> box-muller then mean+std*z
    # (normal_ / randn). unroll is aten's unroll_factor (4 for fp32).
    def __init__(self, kind: str, is_capture: bool, iter_unroll: int, unroll: int = 4):
        self.kind = kind
        self.is_capture = is_capture
        # iter_unroll: grid-stride iterations emitted as a compile-time block (the knob;
        # see _common.choose_iter_unroll). unroll: aten's unroll_factor, i.e. elements per
        # curand4 draw -- structural, 4 for fp32.
        self.iter_unroll = iter_unroll
        self.unroll = unroll

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        seed_val: Int64,
        off_val: Int64,
        intra: Int64,
        argA: Float32,
        argB: Float32,
        argC: Float32,
        num_iters: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mOut, mSeed, mOff, seed_val, off_val, intra, argA, argB, argC, num_iters
        ).launch(grid=[grid_x, 1, 1], block=[_BLOCK, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mSeed: cute.Tensor,
        mOff: cute.Tensor,
        seed_val: Int64,
        off_val: Int64,
        intra: Int64,
        argA: Float32,
        argB: Float32,
        argC: Float32,
        num_iters: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        n = mOut.shape[0]
        idx = Int64(bidx) * _BLOCK + Int64(tidx)
        if cutlass.const_expr(self.is_capture):
            seed = Uint64(mSeed[0])
            off = Int64(mOff[0])
        else:
            seed = Uint64(seed_val)
            off = off_val
        # curand_init(seed, subsequence=idx, offset): offset/4 -> counter.xy (aten's
        # offsets are multiples of 4, so curand's STATE = offset & 3 is 0), and the
        # subsequence goes to counter.zw.
        off4 = (off + intra) >> 2
        k0 = Uint32(seed & 0xFFFFFFFF)
        k1 = Uint32(seed >> 32)
        c2 = Uint32(Uint64(idx) & 0xFFFFFFFF)
        c3 = Uint32(Uint64(idx) >> 32)
        # Threads-per-grid: the gap between the elements one thread's unroll slots cover,
        # and the grid-stride step per iteration is that times unroll.
        tpg = Int64(gdim) * _BLOCK
        stride = tpg * self.unroll

        # HYBRID grid-stride loop: `iter_unroll` iterations are emitted as a compile-time
        # block (giving the scheduler independent philox chains to overlap), and the
        # leftover iterations run one at a time in a dynamic epilogue.
        #
        # iter_unroll is a KNOB with a measured per-arch default, but crucially it is NEVER
        # derived from numel. Baking the trip count itself made compile time scale with
        # tensor size (0.33s at 1 iteration, 14.6s at 222) and produced one kernel per
        # distinct count. A knob keeps compile time and kernel count bounded --
        # O(kind x dtype x capture x unroll) -- while recovering the ILP a fully dynamic
        # loop gives up. See _common.choose_iter_unroll for the sweep data.
        full = num_iters // Int32(self.iter_unroll)
        for blk in cutlass.range(full):
            base_it = blk * Int32(self.iter_unroll)
            for u in cutlass.range_constexpr(self.iter_unroll):
                self._one_iter(
                    mOut,
                    n,
                    idx,
                    tpg,
                    stride,
                    off4,
                    base_it + u,
                    c2,
                    c3,
                    k0,
                    k1,
                    argA,
                    argB,
                    argC,
                )
        for it in cutlass.range(full * Int32(self.iter_unroll), num_iters):
            self._one_iter(
                mOut, n, idx, tpg, stride, off4, it, c2, c3, k0, k1, argA, argB, argC
            )

    @cute.jit
    def _one_iter(
        self,
        mOut: cute.Tensor,
        n,
        idx: Int64,
        tpg: Int64,
        stride: Int64,
        off4: Int64,
        it,
        c2: Uint32,
        c3: Uint32,
        k0: Uint32,
        k1: Uint32,
        argA: Float32,
        argB: Float32,
        argC: Float32,
    ):
        # ONE grid-stride iteration: one curand4 draw -> `unroll` strided elements. Shared
        # by the unrolled main body and the epilogue so there is a single implementation of
        # the counter mapping and the transforms.
        ctr = Uint64(off4 + Int64(it))
        u0, u1, u2, u3 = _philox4(
            Uint32(ctr & 0xFFFFFFFF), Uint32(ctr >> 32), c2, c3, k0, k1
        )
        base = idx + Int64(it) * stride
        if cutlass.const_expr(self.kind == "uniform"):
            # aten: value = x*(to-from) + from in accscalar_t, THEN the bound reversal
            # `value == to ? from : value` (pytorch#16706). curand's uniform is (0, 1], so
            # without that fixup an exact `to` leaks through: measured 1 element in 67M at
            # n=2^26. argC is `to`, passed rather than recomputed so the comparison sees the
            # same float the host derived.
            v0 = _uniform(u0) * argA + argB
            v1 = _uniform(u1) * argA + argB
            v2 = _uniform(u2) * argA + argB
            v3 = _uniform(u3) * argA + argB
            v0 = argB if v0 == argC else v0
            v1 = argB if v1 == argC else v1
            v2 = argB if v2 == argC else v2
            v3 = argB if v3 == argC else v3
        else:
            # curand's _curand_box_muller on the pairs (u0,u1) and (u2,u3), exactly as
            # curand_box_muller4 does. Three details matter for bit-exactness:
            #   * the ANGLE uses CURAND_2POW32_INV_2PI = 2^-32 * float32(2pi) as a single
            #     folded scale (v = y*S + S/2), NOT a uniform times 2pi -- it rounds
            #     differently;
            #   * the pair is (result.x, result.y) = __sincosf(v, &x, &y), i.e. SIN first
            #     then cos (verified by reproducing philox in Python against aten);
            #   * on DEVICE curand uses __sincosf, the FAST-MATH intrinsic, so sin/cos take
            #     fastmath=True -- the accurate versions differ by ~1e-4, far above a ULP.
            s0 = cute.math.sqrt(Float32(-2.0) * cute.math.log(_uniform(u0)))
            a0 = Float32(u1) * Float32(_INV_2PI) + Float32(_INV_2PI_HALF)
            s1 = cute.math.sqrt(Float32(-2.0) * cute.math.log(_uniform(u2)))
            a1 = Float32(u3) * Float32(_INV_2PI) + Float32(_INV_2PI_HALF)
            v0 = cute.math.sin(a0, fastmath=True) * s0 * argA + argB
            v1 = cute.math.cos(a0, fastmath=True) * s0 * argA + argB
            v2 = cute.math.sin(a1, fastmath=True) * s1 * argA + argB
            v3 = cute.math.cos(a1, fastmath=True) * s1 * argA + argB
        # STRIDED writes: slot ii of the draw lands tpg apart, not adjacent.
        if base < n:
            mOut[base] = mOut.element_type(v0)
        if base + tpg < n:
            mOut[base + tpg] = mOut.element_type(v1)
        if base + 2 * tpg < n:
            mOut[base + 2 * tpg] = mOut.element_type(v2)
        if base + 3 * tpg < n:
            mOut[base + 3 * tpg] = mOut.element_type(v3)


def _log_key(kind, dtype, is_capture, iter_unroll):
    return f"{kind} dtype={dtype} capture={is_capture} iter_unroll={iter_unroll}"


# The kernel count is deliberately O(kind x dtype x capture) -- 4 kernels for the two
# distributions in eager + capture -- and NOT a function of tensor size. num_iters is a
# RUNTIME argument: baking it made the grid-stride loop a constexpr unroll, so compile time
# grew linearly with numel (measured cold: 0.33s at 1 iteration, 3.2s at 56, 14.6s at 222 --
# i.e. a 256M-element randn paid ~15s of nvrtc). A dynamic loop compiles once and serves
# every size.
@instrumented_cutedsl_cache("aten::distribution", key_fn=_log_key)
def _compile(kind: str, dtype, is_capture: bool, iter_unroll: int):
    out_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(divisibility=1),), stride=(1,), assumed_align=16
    )
    st = cute.runtime.make_fake_tensor(Int64, (1,), stride=(1,), assumed_align=8)
    st2 = cute.runtime.make_fake_tensor(Int64, (1,), stride=(1,), assumed_align=8)
    return cute.compile(
        _Distribution(kind, is_capture, iter_unroll),
        out_fake,
        st,
        st2,
        Int64(0),
        Int64(0),
        Int64(0),
        Float32(1.0),
        Float32(0.0),
        Float32(1.0),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fill_random(
    out: torch.Tensor,
    kind: str,
    a: float,
    b: float,
    c: float = 0.0,
    iter_unroll: int | None = None,
) -> torch.Tensor:
    """Fill `out` (contiguous, flat-viewable) with the given distribution.

    uniform: value = u*a + b, so a = to-from and b = from; c is `to`, the excluded bound
             that aten maps back to `from` (see the kernel's bound-reversal note).
    normal:  value = z*a + b, so a = std and b = mean; c is unused.
    """
    from ._common import launch_plan, philox_args, state_dummy

    # None -> the measured per-arch default; the autotuner passes an explicit rung. It is
    # part of the compile key, so a sweep produces distinct kernels rather than silently
    # reusing whichever was compiled first.
    if iter_unroll is None:
        iter_unroll = choose_iter_unroll()
    n = out.numel()
    dev = (
        out.device.index
        if out.device.index is not None
        else torch.cuda.current_device()
    )
    grid, counter_offset, num_iters = launch_plan(n, dev, 4)
    seed_t, offset_t, intra = philox_args(dev, counter_offset)
    is_capture = seed_t.is_cuda
    if is_capture:
        st, st2, seed_val, off_val = seed_t, offset_t, 0, 0
    else:
        st = st2 = state_dummy(dev)
        seed_val, off_val = seed_t.item(), offset_t.item()
    _compile(kind, _L.torch2cute[out.dtype], is_capture, iter_unroll)(
        out.view(-1), st, st2, seed_val, off_val, intra, a, b, c, num_iters, grid
    )
    return out
