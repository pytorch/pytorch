r"""Philox distributions with ATen counter order and generator advancement."""

from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from ...cutedsl import hw_caps as _hw, launch as _L
from ...cutedsl.dtypes import torch2cute
from ...cutedsl.types import Index, NumericType
from ...utils.device import check_current
from ._common import _BLOCK


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]


_UNROLL = 4

_PHILOX_10A = 0x9E3779B9
_PHILOX_10B = 0xBB67AE85
_PHILOX_SA = 0xD2511F53
_PHILOX_SB = 0xCD9E8D57

# Match curand_uniform4 rounding into (0, 1], including the half-unit offset.
_INV32 = 2.0**-32
_INV33 = 2.0**-33


# Use curand's pre-rounded angle scale; multiplying a uniform by 2*pi changes rounding.
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
    for round in cutlass.range_constexpr(10):
        lo0, hi0 = _mulhilo(Uint32(_PHILOX_SA), c0)
        lo1, hi1 = _mulhilo(Uint32(_PHILOX_SB), c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if cutlass.const_expr(round < 9):
            k0 = k0 + Uint32(_PHILOX_10A)
            k1 = k1 + Uint32(_PHILOX_10B)
    return c0, c1, c2, c3


@cute.jit
def _uniform(x: Uint32) -> Float32:
    return Float32(x) * Float32(_INV32) + Float32(_INV33)


class _Distribution:
    # Each curand4 draw produces four FP32 elements.

    def __init__(self, kind: str, is_capture: bool) -> None:
        self.kind: str = kind
        self.is_capture: bool = is_capture

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
    ) -> None:
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
    ) -> None:
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
        # Map subsequence to counter.zw and offset/4 to counter.xy in curand order.
        off4 = Int64((Uint64(off) + Uint64(intra)) >> 2)
        k0 = Uint32(seed & 0xFFFFFFFF)
        k1 = Uint32(seed >> 32)
        c2 = Uint32(Uint64(idx) & 0xFFFFFFFF)
        c3 = Uint32(Uint64(idx) >> 32)
        # Each thread writes four slots separated by the threads-per-grid count.
        tpg = Int64(gdim) * _BLOCK
        stride = tpg * 4

        state = (c2, c3, k0, k1)
        params = (argA, argB, argC)
        full = num_iters // Int32(_UNROLL)
        for blk in cutlass.range(full):
            base_it = blk * Int32(_UNROLL)
            for u in cutlass.range_constexpr(_UNROLL):
                self._one_iter(
                    mOut, n, idx, tpg, stride, off4, base_it + u, state, params
                )
        for it in cutlass.range(full * Int32(_UNROLL), num_iters):
            self._one_iter(mOut, n, idx, tpg, stride, off4, it, state, params)

    @cute.jit
    def _one_iter(
        self,
        mOut: cute.Tensor,
        n: Index,
        idx: Int64,
        tpg: Int64,
        stride: Int64,
        off4: Int64,
        it: Index,
        state: tuple[Uint32, Uint32, Uint32, Uint32],
        params: tuple[Float32, Float32, Float32],
    ) -> None:
        c2, c3, k0, k1 = state
        argA, argB, argC = params
        ctr = Uint64(off4 + Int64(it))
        uniforms = _philox4(Uint32(ctr & 0xFFFFFFFF), Uint32(ctr >> 32), c2, c3, k0, k1)
        base = idx + Int64(it) * stride
        values = []
        if cutlass.const_expr(self.kind == "uniform"):
            for slot in cutlass.range_constexpr(4):
                value = _uniform(uniforms[slot]) * argA + argB
                values.append(argB if value == argC else value)
        else:
            for pair in cutlass.range_constexpr(2):
                scale = cute.math.sqrt(
                    Float32(-2.0) * cute.math.log(_uniform(uniforms[2 * pair]))
                )
                angle = Float32(uniforms[2 * pair + 1]) * Float32(_INV_2PI) + Float32(
                    _INV_2PI_HALF
                )
                values.append(cute.math.sin(angle, fastmath=True) * scale * argA + argB)
                values.append(cute.math.cos(angle, fastmath=True) * scale * argA + argB)
        for slot in cutlass.range_constexpr(4):
            index = base + slot * tpg
            if index < n:
                mOut[index] = mOut.element_type(values[slot])


def _log_key(
    kind: str,
    dtype: NumericType,
    is_capture: bool,
    device: int,
    alignment: int,
    arch: tuple[int, int],
) -> str:
    return f"{kind} dtype={dtype} capture={is_capture} device={device} alignment={alignment} arch={arch}"


@instrumented_cutedsl_cache(
    lambda kind, *args, **kwargs: f"aten::{kind}_", key_fn=_log_key
)
def _compile(
    kind: str,
    dtype: NumericType,
    is_capture: bool,
    device: int,
    alignment: int,
    arch: tuple[int, int],
) -> "Function":
    out_fake = _L.fake_compact(dtype, (_L.sym(),), align=alignment)
    st = _L.fake_compact(Int64, (1,), align=8)
    st2 = _L.fake_compact(Int64, (1,), align=8)
    return _L.compile_kernel(
        _Distribution(kind, is_capture),
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
        options=f"--gpu-arch=sm_{arch[0]}{arch[1]}",
    )


def fill_random(
    out: torch.Tensor,
    kind: str,
    a: float,
    b: float,
    c: float = 0.0,
) -> torch.Tensor:
    r"""Fill a contiguous output using the default generator's Philox state."""
    from ._common import launch_plan, philox_args, state_dummy

    n = out.numel()
    dev = (
        out.device.index
        if out.device.index is not None
        else torch.cuda.current_device()
    )
    check_current(dev)
    alignment = _L.supported_alignment(out, 16)
    arch = _hw.caps(out.device).cc
    grid, counter_offset, num_iters = launch_plan(n, dev)
    is_capture = torch.cuda.is_current_stream_capturing()
    target = out.view(-1)
    compiled = _compile(kind, torch2cute[out.dtype], is_capture, dev, alignment, arch)
    dummy = None if is_capture else state_dummy(dev)
    seed_t, offset_t, intra = philox_args(dev, counter_offset)
    if is_capture:
        st, st2, seed_val, off_val = seed_t, offset_t, 0, 0
    else:
        st = st2 = dummy
        seed_val, off_val = seed_t.item(), offset_t.item()
    compiled(target, st, st2, seed_val, off_val, intra, a, b, c, num_iters, grid)
    return out
