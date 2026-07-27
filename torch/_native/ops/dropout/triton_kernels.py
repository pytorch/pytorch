import triton.language as tl

import torch
from torch._native.instrumentation import instrumented_triton_cache

from ._common import _BLOCK, keep_prob_and_scale, launch_plan, philox_args


# Exact float32 values of curand's CURAND_2POW32_INV and CURAND_2POW32_INV/2.
_INV32 = tl.constexpr(2.0**-32)
_INV33 = tl.constexpr(2.0**-33)


def _dropout_log_key(*args, **kwargs) -> str:
    return f"fused_dropout n={args[7]} capture={kwargs['IS_CAPTURE']}"


@instrumented_triton_cache("aten::native_dropout", key_fn=_dropout_log_key)
def _dropout_kernel(
    X_ptr,
    OUT_ptr,
    MASK_ptr,
    SEED,
    OFF,
    intra,
    stride,
    n,
    num_iters,
    P_KEEP,
    SCALE,
    BLOCK: tl.constexpr,
    IS_CAPTURE: tl.constexpr,
):
    # Bit-exact replica of aten's fused_dropout_kernel_vec (VEC=4):
    # counter = offset/4 + iteration, subsequence = global thread id, each
    # curand_uniform4 covers 4 contiguous elements. Mirrors PhiloxCudaState:
    # under capture SEED/OFF are pointers to the generator's extragraph state
    # (DevState, loaded at run time so replays see fresh values); in eager
    # they are int64 scalar arguments (HostState).
    pid = tl.program_id(0)
    tid = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)

    if IS_CAPTURE:
        seed = tl.load(SEED)
        off = tl.load(OFF)
    else:
        seed = SEED
        off = OFF
    off4 = (off + intra) >> 2

    c2 = (tid & 0xFFFFFFFF).to(tl.uint32)
    c3 = ((tid >> 32) & 0xFFFFFFFF).to(tl.uint32)

    for it in range(num_iters):
        ctr = tl.zeros([BLOCK], tl.int64) + (off4 + it)
        c0 = (ctr & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((ctr >> 32) & 0xFFFFFFFF).to(tl.uint32)
        u0, u1, u2, u3 = tl.philox(seed, c0, c1, c2, c3)

        base = tid * 4 + it * stride
        inb = base < n
        k0 = tl.fma(u0.to(tl.float32), _INV32, _INV33) < P_KEEP
        k1 = tl.fma(u1.to(tl.float32), _INV32, _INV33) < P_KEEP
        k2 = tl.fma(u2.to(tl.float32), _INV32, _INV33) < P_KEEP
        k3 = tl.fma(u3.to(tl.float32), _INV32, _INV33) < P_KEEP

        x0 = tl.load(X_ptr + base + 0, mask=inb)
        x1 = tl.load(X_ptr + base + 1, mask=inb)
        x2 = tl.load(X_ptr + base + 2, mask=inb)
        x3 = tl.load(X_ptr + base + 3, mask=inb)
        tl.store(OUT_ptr + base + 0, x0 * k0.to(tl.float32) * SCALE, mask=inb)
        tl.store(OUT_ptr + base + 1, x1 * k1.to(tl.float32) * SCALE, mask=inb)
        tl.store(OUT_ptr + base + 2, x2 * k2.to(tl.float32) * SCALE, mask=inb)
        tl.store(OUT_ptr + base + 3, x3 * k3.to(tl.float32) * SCALE, mask=inb)
        tl.store(MASK_ptr + base + 0, k0, mask=inb)
        tl.store(MASK_ptr + base + 1, k1, mask=inb)
        tl.store(MASK_ptr + base + 2, k2, mask=inb)
        tl.store(MASK_ptr + base + 3, k3, mask=inb)


def dropout_fwd(x: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.numel()
    grid, counter_offset, num_iters = launch_plan(n, x.device.index or 0)
    p_keep, scale = keep_prob_and_scale(p)
    seed_t, offset_t, intra = philox_args(x, counter_offset)
    is_capture = seed_t.is_cuda
    out = torch.empty_like(x)
    mask = torch.empty_like(x, dtype=torch.bool)
    stride = grid * _BLOCK * 4
    _dropout_kernel[(grid,)](
        x,
        out,
        mask,
        seed_t if is_capture else seed_t.item(),
        offset_t if is_capture else offset_t.item(),
        intra,
        stride,
        n,
        num_iters,
        p_keep,
        scale,
        BLOCK=_BLOCK,
        IS_CAPTURE=is_capture,
    )
    return out, mask
