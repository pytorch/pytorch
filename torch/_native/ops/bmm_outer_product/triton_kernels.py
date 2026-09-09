import functools

import triton

import torch
from torch._native.instrumentation import instrument_triton_kernel

from ...triton import ConstTensorWrapper

# Kernel body is shared with the AOT export path: aot_kernel.py carries
# the @triton.jit function (torch-free, loadable by triton.tools.compile);
# here we only add the runtime instrumentation wrapper.
from .aot_kernel import _bmm_outer_product_aot_kernel


# Pin Triton's current default so the launch and its safety check use the
# same number of warps.
_TRITON_DEFAULT_NUM_WARPS = 4


def _bmm_log_key(a, b, out, B, M, N, *strides, BLOCK_M, BLOCK_N, num_warps) -> str:
    # Receives the kernel's launch args; BLOCK_M/BLOCK_N are the constexprs
    # that (with shapes/dtype) form the Triton compile key.
    return (
        f"bmm_outer B={B} M={M} N={N} {a.dtype} "
        f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} num_warps={num_warps}"
    )


_bmm_outer_product_kernel = instrument_triton_kernel("aten::bmm", key_fn=_bmm_log_key)(
    _bmm_outer_product_aot_kernel
)


def _pick_block_sizes(m: int, n: int) -> tuple[int, int]:
    """I swept over some shapes and in the future we should figure out @autotune story"""
    if m <= 32:
        block_m = triton.next_power_of_2(m)
    elif m <= 96:
        block_m = 32
    elif m <= 192:
        block_m = 64
    else:
        block_m = 128
    return block_m, min(triton.next_power_of_2(n), 128)


@functools.lru_cache(maxsize=1024)
def _bmm_outer_product_launch_config(
    batch: int, m: int, n: int
) -> tuple[int, int, int]:
    """Return the 1D grid size and block sizes used to launch the kernel.

    The grid has one entry for every (batch, M tile, N tile). Sharing this
    calculation with the safety guard keeps the checked and launched grids identical.
    """
    block_m, block_n = _pick_block_sizes(m, n)
    grid_size = batch * triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    return grid_size, block_m, block_n


def bmm_outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, _ = a.shape
    N = b.shape[2]

    out = torch.empty(B, M, N, dtype=a.dtype, device=a.device)

    grid_size, BLOCK_M, BLOCK_N = _bmm_outer_product_launch_config(B, M, N)

    # a and b are read-only inputs; wrap them so a copy-on-write tensor is read
    # through const_data_ptr() and not materialized. out is written directly.
    _bmm_outer_product_kernel[(grid_size,)](
        ConstTensorWrapper(a),
        ConstTensorWrapper(b),
        out,
        B,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_TRITON_DEFAULT_NUM_WARPS,
    )
    return out
