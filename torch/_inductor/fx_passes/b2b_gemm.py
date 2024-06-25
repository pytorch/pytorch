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


def b2b_gemm_grid(m, n, meta):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (
        ceildiv(m, meta["ROW_BLOCK_SIZE"]) * ceildiv(n, meta["COL_BLOCK_SIZE"]),
        1,
        1,
    )


b2b_gemm_template = TritonTemplate(
    name="b2b_gemm",
    grid=b2b_gemm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C")}}


    # B2B_GEMM_TRITON_ENTRANCE
    # dram load/store estimations
    #   (A @ B) @ C
    #   M * N, N * O, O * P
    #   baseline
    #     load = M * N + N * O + M * O + O * P
    #     store = M * O + M * P
    #   gemm
    #     load = M * N + M / m * (N * O + O * P)
    #     store = M * P
    M = {{size("A", 0)}}
    # N = {{size("A", 1)}}
    O = {{size("C", 0)}}
    # P = {{size("C", 1)}}

    stride_am = {{stride("A", 0)}}
    stride_an = {{stride("A", 1)}}
    stride_bn = {{stride("B", 0)}}
    stride_bo = {{stride("B", 1)}}
    stride_co = {{stride("C", 0)}}
    stride_cp = {{stride("C", 1)}}

    # A's row block for this thread
    row_block_id = tl.program_id(axis=0)

    # divide B's columns (and C's rows)
    num_col_block = tl.cdiv(O, COL_BLOCK_SIZE)

    # offsets (TODO: handle the non-divisible case)
    offs_row = row_block_id * ROW_BLOCK_SIZE + tl.arange(0, ROW_BLOCK_SIZE)
    offs_col = tl.arange(0, COL_BLOCK_SIZE)  # to be updated in the loop

    # accumulator
    acc = tl.zeros((ROW_BLOCK_SIZE, P), dtype=tl.float16)

    a_ptrs = A + (offs_row[:, None] * stride_am + tl.arange(0, N)[None, :] * stride_an)
    a = tl.load(a_ptrs)

    for _ in range(num_col_block):

        b_ptrs = B + (tl.arange(0, N)[:, None] * stride_bn + offs_col[None, :] * stride_bo)
        b = tl.load(b_ptrs)

        c_ptrs = C + (offs_col[:, None] * stride_co + tl.arange(0, P)[None, :] * stride_cp)
        c = tl.load(c_ptrs)

        # computation (TODO: floating point errors)
        acc += tl.dot(tl.dot(a, b, out_dtype=tl.float16), c, out_dtype=tl.float16)

        # update offsets
        offs_col += COL_BLOCK_SIZE

    # store
    idx_m = offs_row[:, None]
    idx_p = tl.arange(0, P)[None, :]
    mask = (idx_m < M) & (idx_p < P)

    {{store_output(("idx_m", "idx_p"), "acc", "mask")}}
""",
)


B2B_GEMM_PASS = PatternMatcherPass(
    prevent_match_across_mutations=True,
    pass_name="b2b_gemm_pass",
)


def can_apply_b2b_gemm(
    mat1: torch.fx.Node, mat2: torch.fx.Node, mat3: torch.fx.Node
) -> bool:
    if not (("val" in mat1.meta) and ("val" in mat2.meta) and ("val" in mat3.meta)):
        return False
    mat1 = mat1.meta["val"]
    mat2 = mat2.meta["val"]
    mat3 = mat3.meta["val"]
    if not (mat1.is_cuda and mat2.is_cuda and mat3.is_cuda):
        return False
    if not (
        (len(mat1.shape) == 2) and (len(mat2.shape) == 2) and (len(mat3.shape) == 2)
    ):
        return False
    if not ((mat1.shape[1] == mat2.shape[0]) and (mat2.shape[1] == mat3.shape[0])):
        return False
    # TODO: change to a real-check for size restrictions (may consider hardware limit?)
    m, n, o, p = mat1.shape[0], mat1.shape[1], mat3.shape[0], mat3.shape[1]
    m_ok = (m % 128 == 0) and (m > 128)
    n_ok = n == 32
    o_ok = (o % 128 == 0) and (o > 128)
    p_ok = p == 32
    return m_ok and n_ok and o_ok and p_ok


def tuned_b2b_gemm(mat1, mat2, mat3, *, layout=None):
    layout = FixedLayout(
        mat1.get_device(), mat1.get_dtype(), [mat1.shape[0], mat3.shape[1]]
    )
    choices: list[TritonTemplateCaller] = []
    # TODO: change N and P to non-constexpr
    # Note: the only reason why N and P are hardcoded is because in Triton tl.arange(0, N) only works for tl.constexpr
    # TODO: add more configs for tuning (shall I also tune num_stages and num_warps?)
    for config in [
        {
            "ROW_BLOCK_SIZE": 32,
            "COL_BLOCK_SIZE": 32,
            "num_stages": 2,
            "num_warps": 4,
            "N": 32,
            "P": 32,
        },
        {
            "ROW_BLOCK_SIZE": 32,
            "COL_BLOCK_SIZE": 128,
            "num_stages": 2,
            "num_warps": 4,
            "N": 32,
            "P": 32,
        },
        {
            "ROW_BLOCK_SIZE": 128,
            "COL_BLOCK_SIZE": 32,
            "num_stages": 2,
            "num_warps": 4,
            "N": 32,
            "P": 32,
        },
        {
            "ROW_BLOCK_SIZE": 128,
            "COL_BLOCK_SIZE": 128,
            "num_stages": 2,
            "num_warps": 4,
            "N": 32,
            "P": 32,
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
# TODO: match more cases such as bmm and addmm
@register_graph_pattern(
    CallFunction(aten.mm, CallFunction(aten.mm, Arg(), Arg()), Arg()),
    pass_dict=B2B_GEMM_PASS,
)
def b2b_gemm(
    match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node, mat3: torch.fx.Node
) -> None:
    if can_apply_b2b_gemm(mat1, mat2, mat3):
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
