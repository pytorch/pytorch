import cutlass
import numpy as np
import pandas as pd
import triton
import triton.language as tl

import torch


dtype = torch.float32
device = "cuda"
loss = torch.nn.MSELoss()


def cutlass_mm(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    m, n = a.shape[0], b.shape[1]
    d = torch.empty((m, n), dtype=a.dtype, device=a.device)

    plan = cutlass.op.Gemm(element=torch.float32, layout=cutlass.LayoutType.RowMajor)
    plan.math_operation = cutlass.MathOperation.multiply_add_fast_f32

    alpha = 1
    beta = 0
    plan.run(a, b, d, d, alpha, beta, print_module=False)

    return d


@triton.jit
def triton_mm_kernel(
    use_3xtf32,
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if use_3xtf32:
            accumulator = tl.dot(a, b, accumulator, input_precision="tf32x3")
        else:
            accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_mm(a, b, use_3xtf32):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    triton_mm_kernel[grid](
        use_3xtf32,
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        num_warps=8,
        num_stages=3,
    )
    return c


torch.manual_seed(1234)

dims = []
cutlass_3xtf32_loss = []
triton_3xtf32_loss = []
triton_ieee_loss = []
for m in range(256, 4096, 128):
    n = k = m

    a = torch.ones((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)

    allow_tf32_saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    d_ref = torch.mm(a, b)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32_saved

    # d_ref = torch.from_numpy(np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy())).to(
    #    device
    # )

    d_cutlass_3xtf32 = cutlass_mm(a, b)
    d_triton_3xtf32 = triton_mm(a, b, True)
    d_triton_ieee = triton_mm(a, b, False)

    dims.append(m)
    cutlass_3xtf32_loss.append(loss(d_cutlass_3xtf32, d_ref).item())
    triton_3xtf32_loss.append(loss(d_triton_3xtf32, d_ref).item())
    triton_ieee_loss.append(loss(d_triton_ieee, d_ref).item())

df = pd.DataFrame(
    {
        "dims": dims,
        "CUTLASS 3xTF32 loss": cutlass_3xtf32_loss,
        "Triton 3xTF32 loss": triton_3xtf32_loss,
        "Triton IEEE loss": triton_ieee_loss,
    }
)
print(df)
