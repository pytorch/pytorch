import logging
from typing import List

from ..select_algorithm import autotune_select_algorithm, ChoiceCaller, TritonTemplate
from .mm_common import mm_args, int8_mm_configs, mm_grid, mm_options

log = logging.getLogger(__name__)

int_mm_mul_template = TritonTemplate(
    name="int_mm_mul",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B", "S1", "S2" )}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_s1m = {{stride("S1", 0)}}
    stride_s2n = {{stride("S2", 1)}}

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    a_block_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        a = tl.load(a_block_ptr) #, boundary_check=(0, 1))
        b = tl.load(b_block_ptr) #, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        acc += tl.dot(a, b)
        # Advance the block pointer to the next K block.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    s1_ptrs = S1 + offs_m[:, None] * stride_s1m
    s1 = tl.load(s1_ptrs)
    acc = acc.to(tl.float32) * s1
    s2_ptrs = S2 + offs_n[None, :] * stride_s2n
    s2 = tl.load(s2_ptrs)
    acc = acc * s2

    idx_m = offs_m[:, None]
    idx_n = offs_n[None, :]
    mask = (idx_m < M) & (idx_n < N)
    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)


def tuned_int_mm_mul(mat1, mat2, mat3, mat4, *, layout=None):
    out_dtype = mat3.get_dtype()
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=None, out_dtype=out_dtype)
    choices: List[ChoiceCaller] = []
    for config in int8_mm_configs(m, n, k):
        int_mm_mul_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2, mat3, mat4),
            layout=layout,
            **dict(mm_options(config, k, layout),**{"ACC_TYPE": "tl.int32"}),
        )
    return autotune_select_algorithm("int_mm_mul", choices, [mat1, mat2, mat3, mat4], layout)
