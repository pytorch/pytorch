import functools

import triton
import triton.language as tl

import torch
from torch._native.instrumentation import instrumented_triton_cache

from ...triton import ConstTensorWrapper


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


@instrumented_triton_cache("aten::bmm", key_fn=_bmm_log_key)
def _bmm_outer_product_aot_kernel(
    A_ptr,
    B_ptr,
    OUT_ptr,
    B_dim,
    M,
    N,
    stride_ab,
    stride_am,
    stride_bb,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The program id is promoted to int64 once, so every index derived from it
    # (batch, tile, row and column offsets) is 64-bit. Program ids and the
    # int32-range strides are i32, and both pid_b * stride_ob (once
    # (batch - 1) * M * N > INT32_MAX, e.g. (512, 8209, 512)) and
    # pid_m * BLOCK_M (once M > INT32_MAX) used to wrap and write the tail
    # of the output gigabytes before its buffer.
    pid = tl.program_id(0).to(tl.int64)

    # Widen before cdiv adds BLOCK_* - 1, including when a dimension is INT32_MAX.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rm < M
    mask_n = rn < N

    a = tl.load(A_ptr + pid_b * stride_ab + rm * stride_am, mask=mask_m, other=0.0)
    b = tl.load(B_ptr + pid_b * stride_bb + rn * stride_bn, mask=mask_n, other=0.0)

    out = a[:, None] * b[None, :]

    mask = mask_m[:, None] & mask_n[None, :]  # pyrefly: ignore[bad-index]
    tl.store(
        OUT_ptr + pid_b * stride_ob + rm[:, None] * stride_om + rn[None, :] * stride_on,
        out,
        mask=mask,
    )


_bmm_outer_product_kernel = _bmm_outer_product_aot_kernel


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


# fp16 is absent on purpose: the declaration's _DTYPES leaves it to the JIT override,
# so a key here would be a point no grid can reach.
_TL_DTYPES = {"float32": "fp32", "bfloat16": "bf16"}
_DTYPE_SHORT = {"float32": "f32", "bfloat16": "bf16"}


def build(spec: dict) -> dict:
    """One spec point -> a Triton AOT compile request + sidecar. Innermost strides are
    baked to constexpr 1, so only inner-contiguous layouts are served."""
    dtype = spec["dtype"]
    tl_ty = _TL_DTYPES[dtype]
    bm, bn = int(spec["BLOCK_M"]), int(spec["BLOCK_N"])
    prefix = f"bmm_outer_{_DTYPE_SHORT[dtype]}_bm{bm}_bn{bn}"
    # Parity with the JIT specializer's baked strides and 16B hints: without them the
    # SASS is generically addressed, measured ~7x slower. The ":16" suffix is the
    # toolchain's spelling for a tt.divisibility attr, which the prelude's 16B
    # alignment tests are what make true.
    ptr = f"*{tl_ty}:16"
    signature = ", ".join(
        [ptr, ptr, ptr, "i32", "i32", "i32"]
        + ["i32", "1", "i32", "1", "i32", "i32", "1"]  # am/bn/on baked to 1
        + [str(bm), str(bn)]
    )
    grid_x = f"B_dim*(((M+{bm - 1}LL)/{bm})*((N+{bn - 1}LL)/{bn}))"
    return {
        "kind": "triton",
        "prefix": prefix,
        "fn": _bmm_outer_product_aot_kernel,
        "signature": signature,
        # Evaluated in the generated launcher, over the named scalar args.
        "launch": {"grid_x": grid_x},
        "num_warps": 4,
        # Signature order, constexprs excluded; passed positionally.
        "args": [
            {"name": "a", "kind": "tensor", "read_only": True},
            {"name": "b", "kind": "tensor", "read_only": True},
            {"name": "out", "kind": "tensor"},
            {"name": "B_dim", "kind": "scalar", "ctype": "int32_t"},
            {"name": "M", "kind": "scalar", "ctype": "int32_t"},
            {"name": "N", "kind": "scalar", "ctype": "int32_t"},
            {"name": "stride_ab", "kind": "scalar", "ctype": "int32_t"},
            {"name": "stride_bb", "kind": "scalar", "ctype": "int32_t"},
            {"name": "stride_ob", "kind": "scalar", "ctype": "int32_t"},
            {"name": "stride_om", "kind": "scalar", "ctype": "int32_t"},
        ],
    }
