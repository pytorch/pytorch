# mypy: allow-untyped-defs
import torch
from ..._dynamo.utils import counters
from ..ir import FixedLayout
from ..pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    TritonTemplate,
    TritonTemplateCaller,
)
from ..utils import ceildiv

aten = torch.ops.aten


def b2b_gemm_grid(M, P, meta):
    return (ceildiv(M, meta["BLOCK_SIZE_M"]) * ceildiv(P, meta["BLOCK_SIZE_P"]), 1, 1)


b2b_gemm_template = TritonTemplate(
    name="b2b_gemm",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_TRITON_ENTRANCE

    # dynamic shapes
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    # dynamic strides
    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output block counts
    num_m_block = tl.cdiv(M, BLOCK_SIZE_M)
    num_p_block = tl.cdiv(P, BLOCK_SIZE_P)

    # internal block counts
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # output block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // num_p_block
    p_block_id = pid % num_p_block

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    # main loop
    offs_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))
    for _ in range(num_n_block):
        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_M * BLOCK_SIZE_N
        offs_o = tl.arange(0, BLOCK_SIZE_O)
        acc_bc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
        for __ in range(num_o_block):
            b_mask = (offs_n[:, None] < N) & (offs_o[None, :] < O)
            b_ptrs = B + (offs_n[:, None] * stride_bn + offs_o[None, :] * stride_bo)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_N * BLOCK_SIZE_O
            c_mask = (offs_o[:, None] < O) & (offs_p[None, :] < P)
            c_ptrs = C + (offs_o[:, None] * stride_co + offs_p[None, :] * stride_cp)
            # Note: this load is independent of the outer loop; can we somehow move it?
            c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)  # BLOCK_SIZE_O * BLOCK_SIZE_P
            acc_bc += tl.dot(b, c, out_dtype=tl.float32)
            offs_o += BLOCK_SIZE_O
        acc += tl.dot(a, acc_bc, out_dtype=tl.float32)
        offs_n += BLOCK_SIZE_N

    # type conversion
    acc = acc.to(tl.float16)

    # store preparation
    idx_m = offs_m[:, None]
    idx_p = offs_p[None, :]
    d_mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "d_mask")}}
""",
)


B2B_GEMM_PASS = PatternMatcherPass(
    prevent_match_across_mutations=True,
    pass_name="b2b_gemm_pass",
)


def can_apply_b2b_gemm(match: Match) -> bool:
    if not all("val" in arg.meta for arg in match.args):
        return False
    mats = [arg.meta["val"] for arg in match.args]
    if not all(mat.is_cuda for mat in mats):
        return False
    if not all(len(mat.shape) == 2 for mat in mats):
        return False
    mat1, mat2, mat3 = mats
    if not ((mat1.shape[1] == mat2.shape[0]) and (mat2.shape[1] == mat3.shape[0])):
        return False
    # TODO: change to a real-check for size restrictions (may consider hardware limit?)
    return True


def tuned_b2b_gemm(mat1, mat2, mat3, *, layout=None):
    layout = FixedLayout(
        mat1.get_device(), mat1.get_dtype(), [mat1.shape[0], mat3.shape[1]]
    )
    choices: list[TritonTemplateCaller] = []
    # TODO: add more configs for tuning
    # based on the assumption, when M, O >> N, P the optimization should be effective
    # and it's good to set larger BLOCK_SIZE_M
    for config in [
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_O": 128,
            "BLOCK_SIZE_P": 32,
            "num_stages": 2,
            "num_warps": 4,
        },
    ]:
        b2b_gemm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2, mat3),
            layout=layout,
            **config,
        )
    return autotune_select_algorithm("b2b_gemm", choices, [mat1, mat2, mat3], layout)


# currently it matches ((A @ B) @ C)
# TODO: later will change to matching (A @ B) in (epilogue2 ((epilogue1 (A @ B)) @ C)) and inspecting the graph
# TODO: match more cases such as bmm and addmm, and (A @ (B @ C)), etc.
@register_graph_pattern(
    CallFunction(aten.mm, CallFunction(aten.mm, Arg(), Arg()), Arg()),
    extra_check=can_apply_b2b_gemm,
    pass_dict=B2B_GEMM_PASS,
)
def b2b_gemm(
    match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node, mat3: torch.fx.Node
) -> None:
    counters["inductor"]["b2b_gemm"] += 1
    graph = match.graph
    root_node = match.nodes[-1]
    with graph.inserting_before(root_node):
        tuned_b2b_gemm._inductor_lowering_function = True  # type: ignore[attr-defined]
        replacement = graph.call_function(
            tuned_b2b_gemm, tuple(match.args), match.kwargs
        )
        replacement.meta.update(root_node.meta)
        root_node.replace_all_uses_with(replacement)
    match.erase_nodes(graph)
