import torch

from ..utils import has_triton

if has_triton():
    import triton
    import triton.language as tl

    def init_to_zero(name):
        return lambda nargs: nargs[name].zero_()

    @triton.heuristics(
        {
            "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        }
    )
    @triton.autotune(
        configs=[
            # basic configs for compute-bound matmuls
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
                num_stages=5,
                num_warps=2,
            ),
            # additional configs
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=2,
                num_warps=4,
            ),
            # additional configs for K = 64
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
                num_stages=1,
                num_warps=2,
            ),
        ],
        # + get_configs_io_bound(),
        key=["M", "N", "K"],
        #
        # key=["M", "N", "K"],
        # prune_configs_by={
        #     "early_config_prune": early_config_prune,
        #     "perf_model": estimate_matmul_time,
        #     "top_k": 18,
        # },
    )
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
        bid = tl.program_id(2)
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
        A += bid * M * K
        B += bid * K * N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for k in range(K, 0, -BLOCK_K * SPLIT_K):
            if EVEN_K:
                a = tl.load(A)
                b = tl.load(B)
            else:
                a = tl.load(A, mask=rk[None, :] < k, other=0.0)
                b = tl.load(B, mask=rk[:, None] < k, other=0.0)
            acc += tl.dot(a, b)
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        C += bid * M * N
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        if SPLIT_K == 1:
            tl.store(C, acc, mask=mask)
        else:
            tl.atomic_add(C, acc, mask=mask)

    def bmm_out(a, b, out):
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[2] == b.shape[1], "incompatible dimensions"
        B, M, K = a.shape
        _, _, N = b.shape
        # allocates output
        c = out
        # accumulator types
        ACC_TYPE = (
            tl.float32
            if a.dtype in [torch.float16, torch.bfloat16, torch.float32]
            else tl.int32
        )

        # launch kernel
        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                META["SPLIT_K"],
                B,
            )

        # grid = lambda META: (
        #     triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        #     META["SPLIT_K"],
        #     B,
        # )

        # autotuner = _kernel[grid].kernel
        _kernel[grid](a, b, c, M, N, K, K, 1, N, 1, N, 1, GROUP_M=8, ACC_TYPE=ACC_TYPE)
        # print(autotuner.best_config)
        # print(autotuner.configs_timings)
