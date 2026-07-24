"""Triton bmm outer-product kernel for AOT export.

This file exists separately from triton_kernels.py because
triton.tools.compile takes a kernel FILE path and executes it
standalone to find the JITFunction -- so the file must hold the bare
@triton.jit function without the JIT wrapper's instrumentation
decorator. The JIT wrapper (triton_kernels.py) imports the kernel from
here, so both routes share one body.

``_bmm_outer_product_aot_kernel`` computes ``out[b] = a[b] @ b[b]`` for
the K==1 outer-product case: a is (B, M, 1), b is (B, 1, N), out is
(B, M, N); strides are runtime arguments so non-contiguous inputs work.
Identical math to the JIT kernel minus the instrumentation decorator.

``build(spec)`` is the AOT entry point (see tools/native_aot/export.py,
kind="triton"): each spec point bakes one (dtype, BLOCK_M, BLOCK_N)
specialization; the grid expression is baked into the generated C
entry point in terms of the named integer arguments.
"""

import os

import triton
import triton.language as tl


@triton.jit
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
    pid = tl.program_id(0)

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


_TL_DTYPES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
_DTYPE_SHORT = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}


def build(spec: dict) -> dict:
    """One manifest spec point -> a Triton AOT compile request + sidecar.

    spec: {"dtype": ..., "BLOCK_M": int, "BLOCK_N": int}. Tensor strides
    are runtime i64 args, so one specialization serves all layouts; the
    guard over which (M, N) bucket routes here lives in the manifest
    (per-spec `guard:`), mirroring the JIT wrapper's block-size picker.
    """
    dtype = spec["dtype"]
    tl_ty = _TL_DTYPES[dtype]
    bm, bn = int(spec["BLOCK_M"]), int(spec["BLOCK_N"])
    kind = spec.get("toolchain", "triton")
    prefix = f"bmm_outer_{_DTYPE_SHORT[dtype]}_bm{bm}_bn{bn}"
    if kind == "triton_cubin":
        prefix += "_rc"  # distinct artifacts for the raw-cubin A/B spike
    # Specialization parity with the JIT route: Triton's runtime
    # specializer bakes the innermost (contiguous) strides to constexpr 1
    # and hints 16-byte pointer alignment; without matching this the AOT
    # SASS is generically-addressed and measurably slower (up to ~7x on
    # this memory-bound kernel). The declaration's prelude enforces the
    # corresponding runtime conditions (innermost stride 1, aligned
    # data_ptrs); batch/row strides stay runtime i64.
    ptr = f"*{tl_ty}:16"
    signature = ", ".join(
        [ptr, ptr, ptr, "i32", "i32", "i32"]
        + ["i32", "1", "i32", "1", "i32", "i32", "1"]  # am/bn/on baked to 1
        + [str(bm), str(bn)]
    )
    grid_x = f"B_dim*(((M+{bm - 1})/{bm})*((N+{bn - 1})/{bn}))"
    return {
        "kind": kind,
        "prefix": prefix,
        "kernel_path": os.path.abspath(__file__),
        "kernel_name": "_bmm_outer_product_aot_kernel",
        "signature": signature,
        # triton kind: grid string baked into the C entry point.
        # triton_cubin kind: grid exprs consumed by the generic launcher.
        "grid": f"{grid_x}, 1, 1",
        "launch": {"grid_x": grid_x},
        "num_warps": 4,
        # Flat argument list in signature order (constexprs excluded):
        # the launcher passes data_ptr()/int values positionally.
        # Constexpr-baked strides (am/bn/on = 1) are excluded from the
        # entry-point prototype; the launcher passes only these.
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
