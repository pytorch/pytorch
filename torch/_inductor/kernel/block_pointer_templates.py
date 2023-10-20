import logging
from typing import List

from ..select_algorithm import autotune_select_algorithm, ChoiceCaller, TritonTemplate
from .mm_common import int8_mm_configs, mm_args, mm_grid, mm_options

log = logging.getLogger(__name__)

mm_block_pointer_template = TritonTemplate(
    name="mm_bp",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    # -----------------------------------------------------------
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    a_block_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
                                    order=(1, 0))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
        else:
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)
