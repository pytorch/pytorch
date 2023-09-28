import functools

import torch

from ..lowering import lowerings
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid, mm_options

aten = torch.ops.aten

aten_mm_plus_mm = ExternKernelChoice(
    torch.ops.inductor._mm_plus_mm, "torch::inductor::_mm_plus_mm"
)

mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",
    grid=mm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C", "D")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    # K2 = {{size("C", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_cm = {{stride("C", 0)}}
    stride_ck = {{stride("C", 1)}}
    stride_dk = {{stride("D", 0)}}
    stride_dn = {{stride("D", 1)}}

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

    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0)
    )
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, K),
        strides=(stride_cm, stride_ck),
        offsets=(block_offset_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        base=D,
        shape=(K, N),
        strides=(stride_dk, stride_dn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k1 in range(K, 0, -BLOCK_K):
        # First matmul with A @ B
        if EVEN_K:
            a = tl.load(A_block_ptr, boundary_check=(0))
            b = tl.load(B_block_ptr, boundary_check=(0))
        else:
            a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
            b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    for k2 in range(K, 0, -BLOCK_K):

        # Second matmul with C @ D
        if EVEN_K:
            c = tl.load(C_block_ptr, boundary_check=(0))
            d = tl.load(D_block_ptr, boundary_check=(0))
        else:
            c = tl.load(C_block_ptr, boundary_check=(0, 1), padding_option='zero')
            d = tl.load(D_block_ptr, boundary_check=(0, 1), padding_option='zero')
        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)
        C_block_ptr = tl.advance(C_block_ptr, (0, BLOCK_K))
        D_block_ptr = tl.advance(D_block_ptr, (BLOCK_K, 0))

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)


@functools.lru_cache(None)
def mm_configs():
    import triton

    # List of dictionaries to store the kernel configs. Configs that evaluate to true
    # will be utilised on the target platform
    mm_triton_configs = [
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 3,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 16,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            "num_stages": 1,
            "num_warps": 8,
            "cond": torch.version.hip is None,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16},
            "num_stages": 1,
            "num_warps": 2,
            "cond": True,
        },
    ]

    # Filter out configs in which cond evaluates to true
    # On ROCm convert num_stages to 1 as pipelining provides no benefit
    if torch.version.hip:
        filtered_configs = [
            triton.Config(c["config"], num_stages=1, num_warps=c["num_warps"])
            for c in mm_triton_configs
            if c["cond"]
        ]
    else:
        filtered_configs = [
            triton.Config(
                c["config"], num_stages=c["num_stages"], num_warps=c["num_warps"]
            )
            for c in mm_triton_configs
            if c["cond"]
        ]

    return filtered_configs


def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    """
    Computes mm(mat1, mat2) + mm(mat3, mat4)
    """
    m1, n1, k1, layout1, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    m2, n2, _, layout2, mat3, mat4 = mm_args(mat3, mat4, layout=layout)
    # Optimization is optional, because we can always just not do the fusion
    if (
        m1 * n1 == 0
        or m2 * n2 == 0
        or not V.graph.sizevars.statically_known_list_equals(
            mat1.get_size(), mat3.get_size()
        )
        or not V.graph.sizevars.statically_known_list_equals(
            mat2.get_size(), mat4.get_size()
        )
    ):
        # TODO(jansel): support different K values when this is fixed:
        # https://github.com/openai/triton/issues/967
        if m1 == m2 and n1 == n2:
            V.graph.sizevars.guard_equals(m1, m2)
            V.graph.sizevars.guard_equals(n1, n2)
            return lowerings[aten.addmm](lowerings[aten.mm](mat3, mat4), mat1, mat2)
        return lowerings[aten.add](
            lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4)
        )

    assert layout1 == layout2
    # options to tune from
    choices = (
        [aten_mm_plus_mm.bind((mat1, mat2, mat3, mat4), layout1)]
        if use_aten_gemm_kernels()
        else []
    )
    if use_triton_template(layout1):
        for config in mm_configs():
            # see https://github.com/openai/triton/issues/1298
            # BLOCK_K = K causes llvm error
            if config.kwargs["BLOCK_K"] < k1:
                mm_plus_mm_template.maybe_append_choice(
                    choices,
                    input_nodes=(mat1, mat2, mat3, mat4),
                    layout=layout1,
                    **mm_options(config, k1, layout1),
                )

    return autotune_select_algorithm(
        "mm_plus_mm", choices, [mat1, mat2, mat3, mat4], layout1
    )
