import logging

import torch
from .. import config as inductor_config
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import use_triton_template
from .mm_common import (
    addmm_epilogue,
    int8_mm_configs,
    mm_args,
    mm_configs,
    mm_grid,
    mm_options,
)

log = logging.getLogger(__name__)
aten = torch.ops.aten

mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)
def sdpa_grid(S, H, M, D, meta):
    import triton
    return (triton.cdiv(M, meta["BLOCK_M"]), S*H, 1)
    # return (1, 1, 1)

sdpa_template = TritonTemplate(
    name="sdpa",
    grid=sdpa_grid,
    source=r"""
{{def_kernel("Q", "K", "V")}}
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qk = {{stride("Q", 3)}}
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_kn = {{stride("K", 2)}}
    stride_kk = {{stride("K", 3)}}
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    stride_vk = {{stride("V", 2)}}
    stride_vn = {{stride("V", 3)}}

    # stride_oz = 131072
    # stride_oh = 32768
    # stride_om = 64
    # stride_on = 1
    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    N_CTX = {{size("Q", 2)}}

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # O_block_ptr = tl.make_block_ptr(
    #     base=out_ptr1 + qvk_offset,
    #     shape=(N_CTX, BLOCK_DMODEL),
    #     strides=(stride_om, stride_on),
    #     offsets=(start_m * BLOCK_M, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    lo, hi = 0, N_CTX
    # causal check on every loop iteration can be expensive
    # and peeling the last iteration of the loop does not work well with ptxas
    # so we have a mode to do the causal check in a separate kernel entirely
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # advance block pointers to first iteration of the loop
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        # qk = attention_modification_triton(qk, offs_m[:, None], (start_n + offs_n[None, :]), off_hz)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # -- compute m_ij, p, l_ij
        sm_scale = 1
        qk = qk * sm_scale
        m_ij = tl.max(qk, 1)
        # m_ij = tl.maximum(m_ij, -1e9)
        is_fully_masked = m_ij == float("-inf")

        p = tl.where(is_fully_masked[:, None], 0, tl.math.exp(qk - m_ij[:, None]))
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_i_new)
        beta = tl.math.exp(m_ij - m_i_new)
        l_i = tl.where(is_fully_masked, l_i, l_i * alpha)
        l_i_new = tl.where(is_fully_masked, l_i, l_i + beta * l_ij)
        # scale p
        p_scale = beta / l_i_new
        p = tl.where(is_fully_masked[:, None], p, p * p_scale[:, None])
        # scale acc
        acc_scale = l_i / l_i_new
        acc = tl.where(is_fully_masked[:, None], acc, acc * acc_scale[:, None])
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, l_i)
    # tl.store(m_ptrs, m_i)
    # write back O
    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
    mask = (idx_m != -1) & (idx_d != -1)
    {{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc", "mask")}}
    # tl.store(O_block_ptr, acc.to(tl.float16))
 """
)

aten_mm = ExternKernelChoice(torch.mm, "at::mm_out")


aten_addmm = ExternKernelChoice(torch.addmm, "at::addmm_out")

aten__int_mm = ExternKernelChoice(torch._int_mm, "at::_int_mm")


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if inp.stride(0) == 0 or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


aten_bias_addmm = ExternKernelChoice(bias_addmm, None)


@register_lowering(aten.mm)
def tuned_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

    # options to tune from
    choices = []
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(
                choices,
                (mat1, mat2),
                layout,
                **mm_options(config, k, layout),
            )

    return autotune_select_algorithm("mm", choices, [mat1, mat2], layout)


@register_lowering(aten._int_mm)
def tuned_int_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    choices = [aten__int_mm.bind((mat1, mat2), layout)]
    if use_triton_template(layout, enable_int32=True):
        # TODO: Re-enable eager mode implementation once cuBLAS is fixed
        choices = []
        for config in int8_mm_configs(m, n, k):
            mm_template.maybe_append_choice(
                choices,
                (mat1, mat2),
                layout,
                **mm_options(config, k, layout),
            )
    return autotune_select_algorithm("int_mm", choices, [mat1, mat2], layout)


@register_lowering(aten.addmm)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    ordered_kwargs_for_cpp_kernel = ("beta", "alpha")

    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    if not use_triton_template(layout):
        choices = [
            aten_addmm.bind(
                (inp, mat1, mat2),
                layout,
                ordered_kwargs_for_cpp_kernel,
                alpha=alpha,
                beta=beta,
            )
        ]
        return autotune_select_algorithm("addmm", choices, [inp, mat1, mat2], layout)

    choices = [
        aten_addmm.bind(
            (inp_expanded, mat1, mat2),
            layout,
            ordered_kwargs_for_cpp_kernel,
            alpha=alpha,
            beta=beta,
        )
    ]
    if (
        inp_expanded.get_stride()[0] == 0
        and inp_expanded.get_device().type == "cuda"
        and inductor_config.triton.autotune_cublasLt
    ):
        # unexpand inp to make sure fused addmm from cublasLt is used
        choices.insert(
            0,
            aten_bias_addmm.bind(
                (inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta
            ),
        )

    for config in mm_configs(m, n, k):
        mm_template.maybe_append_choice(
            choices,
            (inp_expanded, mat1, mat2),
            layout,
            **mm_options(config, k, layout),
            prefix_args=1,
            epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
        )

    return autotune_select_algorithm(
        "addmm", choices, [inp_expanded, mat1, mat2], layout
    )
