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
    # TODO: handle the non-divisible case
    M = {{size("A", 0)}}
    N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    P = {{size("C", 1)}}

    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # output (M * P) block ids
    pid = tl.program_id(axis=0)
    m_block_id = pid // (P // BLOCK_SIZE_P)
    p_block_id = pid % (P // BLOCK_SIZE_P)

    # internal block numbers
    num_n_block = tl.cdiv(N, BLOCK_SIZE_N)
    num_o_block = tl.cdiv(O, BLOCK_SIZE_O)

    # accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)

    for n_block_id in range(num_n_block):
        a_ptrs = A + (
            (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am +
            (n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_an
        )
        a = tl.load(a_ptrs)
        for o_block_id in range(num_o_block):
            b_ptrs = B + (
                (n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None] * stride_bn +
                (o_block_id * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O))[None, :] * stride_bo
            )
            b = tl.load(b_ptrs)
            c_ptrs = C + (
                (o_block_id * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O))[:, None] * stride_co +
                (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))[None, :] * stride_cp
            )
            c = tl.load(c_ptrs)
            acc += tl.dot(tl.dot(a, b, out_dtype=tl.float16), c, out_dtype=tl.float16)

    # store
    idx_m = (m_block_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    idx_p = (p_block_id * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P))[None, :]
    mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "mask")}}
""",
)


B2B_GEMM_PASS = PatternMatcherPass(
    prevent_match_across_mutations=True,
    pass_name="b2b_gemm_pass",
)


def can_apply_b2b_gemm(match: Match) -> bool:
    if not all(["val" in arg.meta for arg in match.args]):
        return False
    mats = [arg.meta["val"] for arg in match.args]
    if not all([mat.is_cuda for mat in mats]):
        return False
    if not all([len(mat.shape) == 2 for mat in mats]):
        return False
    mat1, mat2, mat3 = mats
    if not ((mat1.shape[1] == mat2.shape[0]) and (mat2.shape[1] == mat3.shape[0])):
        return False
    # TODO: change to a real-check for size restrictions (may consider hardware limit?)
    m, n, o, p = mat1.shape[0], mat1.shape[1], mat3.shape[0], mat3.shape[1]
    m_ok = (m % 64 == 0) and (m >= 64)
    n_ok = (n % 64 == 0) and (n >= 64)
    o_ok = (o % 64 == 0) and (o >= 64)
    p_ok = (p % 64 == 0) and (p >= 64)
    return m_ok and n_ok and o_ok and p_ok


def tuned_b2b_gemm(mat1, mat2, mat3, *, layout=None):
    layout = FixedLayout(
        mat1.get_device(), mat1.get_dtype(), [mat1.shape[0], mat3.shape[1]]
    )
    choices: list[TritonTemplateCaller] = []
    # TODO: add more configs for tuning (shall I also tune num_stages and num_warps?)
    for config in [
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_O": 64,
            "BLOCK_SIZE_P": 64,
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
