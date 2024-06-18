import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
from .split_cat import construct_pattern_matcher_pass
from ..select_algorithm import TritonTemplate

aten = torch.ops.aten

# TODO: change to triton template
kernel_source = """
@triton.jit
def gemm_kernel(
    # pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    # matrix dimensions
    M,
    N: tl.constexpr,  # TODO: get rid of constexpr?
    O,
    P: tl.constexpr,
    # strides
    stride_am,
    stride_an,
    stride_bn,
    stride_bo,
    stride_co,
    stride_cp,
    stride_dm,
    stride_dp,
    # other parameters
    ROW_BLOCK_SIZE: tl.constexpr,  # row block size for A
    COL_BLOCK_SIZE: tl.constexpr,  # col block size for B (also the row block size for C)
):
    # dram load/store estimations
    #   (A @ B) @ C
    #   M * N, N * O, O * P
    #   baseline
    #     load = M * N + N * O + M * O + O * P
    #     store = M * O + M * P
    #   gemm
    #     load = M * N + M / m * (N * O + O * P)
    #     store = M * P

    # A's row block for this thread
    row_block_id = tl.program_id(axis=0)

    # divide B's columns (and C's rows)
    num_col_block = tl.cdiv(O, COL_BLOCK_SIZE)

    # offsets (TODO: handle the non-divisible case)
    offs_row = row_block_id * ROW_BLOCK_SIZE + tl.arange(0, ROW_BLOCK_SIZE)
    offs_col = tl.arange(0, COL_BLOCK_SIZE)  # to be updated in the loop

    # accumulator
    acc = tl.zeros((ROW_BLOCK_SIZE, P), dtype=tl.float16)

    # A
    a_ptrs = a_ptr + (
        offs_row[:, None] * stride_am + tl.arange(0, N)[None, :] * stride_an
    )

    a = tl.load(a_ptrs)

    for _ in range(num_col_block):

        # B
        b_ptrs = b_ptr + (
            tl.arange(0, N)[:, None] * stride_bn + offs_col[None, :] * stride_bo
        )

        b = tl.load(b_ptrs)

        # C
        c_ptrs = c_ptr + (
            offs_col[:, None] * stride_co + tl.arange(0, P)[None, :] * stride_cp
        )

        c = tl.load(c_ptrs)

        # computation (TODO: floating point errors)
        acc += tl.dot(tl.dot(a, b, out_dtype=tl.float16), c, out_dtype=tl.float16)

        # update offsets
        offs_col += COL_BLOCK_SIZE

    # store
    d_ptrs = d_ptr + (
        offs_row[:, None] * stride_dm + tl.arange(0, P)[None, :] * stride_dp
    )

    tl.store(d_ptrs, acc)
"""

def can_apply_b2b_gemm(mat1, mat2, mat3) -> bool:
    if not(is_node_meta_valid(mat1) and is_node_meta_valid(mat2) and is_node_meta_valid(mat3)):
        return False
    mat1 = mat1.meta["val"]
    mat2 = mat2.meta["val"]
    mat3 = mat3.meta["val"]
    if not (a.is_cuda and b.is_cuda):
        return False
    if len(mat1.shape) != 2 or len(mat2.shape) != 2:
        return False
    # TODO: check for size restrictions
    return True

# currently it matches ((A @ B) @ C)
# later will change to matching (A @ B) in (epilogue2 ((epilogue1 (A @ B)) @ C)) and inspecting the graph
@register_graph_pattern(
    CallFunction(aten.mm, CallFunction(aten.mm, Arg(), Arg()), Arg()),
    pass_dict=construct_pattern_matcher_pass("b2b_gemm_pass"),
)
def b2b_gemm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node, mat3: torch.fx.Node):
    print("B2B-GEMM handler called")
    # if can_apply_b2b_gemm(mat1, mat2, mat3):
    #     counters["inductor"]["b2b_gemm"] += 1
    #     lowering to the B2B_GEMM Triton template (not sure how to do this)
