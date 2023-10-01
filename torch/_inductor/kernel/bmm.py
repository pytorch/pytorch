import torch

from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import ceildiv as cdiv, use_aten_gemm_kernels, use_triton_template

from .mm_common import addmm_epilogue, mm_args, mm_configs, mm_options

aten = torch.ops.aten


def bmm_grid(b, m, n, meta):
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), b, 1)


bmm_template = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # re-order program ID for better L2 performance
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M
    block_offset_m = pid_m * BLOCK_M
    block_offset_n = pid_n * BLOCK_N

    idx_q = tl.program_id(1)  # batch dimension for BMM

    A_block_ptr = tl.make_block_ptr(
        base=A + idx_q * stride_aq,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0) if stride_am >= stride_ak else (0, 1),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B + idx_q * stride_bq,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0) if stride_bk >= stride_bn else (0, 1)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A_block_ptr, boundary_check=(0))
            b = tl.load(B_block_ptr, boundary_check=(0))
        else:
            a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
            b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask")}}
""",
)

aten_bmm = ExternKernelChoice(torch.bmm, "at::bmm_out")
aten_baddbmm = ExternKernelChoice(torch.baddbmm, "at::baddbmm_out")


@register_lowering(aten.bmm)
def tuned_bmm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

    # options to tune from
    choices = [aten_bmm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, k, layout),
            )

    return autotune_select_algorithm("bmm", choices, [mat1, mat2], layout)


# Don't register this since it is slower than decomposing it
# @register_lowering(aten.baddbmm)
def tuned_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    m, n, k, layout, mat1, mat2, inp = mm_args(mat1, mat2, inp, layout=layout)

    # options to tune from
    choices = (
        [aten_baddbmm.bind((inp, mat1, mat2), layout, alpha=alpha, beta=beta)]
        if use_aten_gemm_kernels()
        else []
    )
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(inp, mat1, mat2),
                layout=layout,
                **mm_options(config, k, layout),
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
            )

    return autotune_select_algorithm("baddbmm", choices, [inp, mat1, mat2], layout)
