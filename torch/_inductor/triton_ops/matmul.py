import torch

from ..utils import has_triton

if has_triton():

    import triton
    import triton.language as tl

    from .autotune import mm_autotune, mm_heuristics

    @mm_heuristics()
    @mm_autotune(get_io_bound_configs=True)
    @triton.jit
    def _kernel(
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        allow_tf32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        ACC_TYPE: tl.constexpr,
    ):
        # matrix multiplication
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        # do matrix multiplication
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
        # pointers
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k in range(K, 0, -BLOCK_K * SPLIT_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k, other=0.0)
            acc += tl.dot(a, b, allow_tf32=allow_tf32)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)
        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        if SPLIT_K == 1:
            tl.store(C, acc, mask=mask)
        else:
            tl.atomic_add(C, acc, mask=mask)

    class _matmul_out:
        kernel = _kernel

        @staticmethod
        def _call(a, b, out, allow_tf32=True):
            # handle non-contiguous inputs if necessary
            if a.stride(0) > 1 and a.stride(1) > 1:
                a = a.contiguous()
            if b.stride(0) > 1 and b.stride(1) > 1:
                b = b.contiguous()
            # checks constraints
            assert a.shape[1] == b.shape[0], "incompatible dimensions"
            M, K = a.shape
            _, N = b.shape
            # allocates output
            c = out
            # accumulator types
            ACC_TYPE = (
                tl.float32
                if a.dtype in [torch.float16, torch.bfloat16, torch.float32]
                else tl.int32
            )

            # launch kernel (grid defined as using def instead of lambda to pass `make lint`)
            def grid(META):
                return (
                    triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                    META["SPLIT_K"],
                )

            # grid = lambda META: (
            #     triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            #     META["SPLIT_K"],
            # )
            _kernel[grid](
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
                allow_tf32=allow_tf32,
                GROUP_M=8,
                ACC_TYPE=ACC_TYPE,
            )

        @staticmethod
        def forward(a, b, out, allow_tf32=True):
            return _matmul_out._call(a, b, out, allow_tf32)

    matmul_out = _matmul_out.forward
