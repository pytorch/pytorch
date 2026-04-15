import triton
import triton.language as tl

import torch


@triton.jit
def _bmm_outer_product_kernel(
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


def bmm_outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, M, _ = a.shape
    N = b.shape[2]

    out = torch.empty(B, M, N, dtype=a.dtype, device=a.device)

    BLOCK_M, BLOCK_N = _pick_block_sizes(M, N)

    _bmm_outer_product_kernel[(B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)](
        a,
        b,
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
    )
    return out
