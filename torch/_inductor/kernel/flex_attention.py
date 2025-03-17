# mypy: allow-untyped-defs
"""Triton Implementation of the flex_attention Kernel"""

import copy
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Optional, Union

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges

from .. import config
from ..ir import (
    Buffer,
    ComputedBuffer,
    ExternKernel,
    FixedLayout,
    FlexibleLayout,
    get_fill_order,
    InputBuffer,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    Scatter,
    StorageBox,
    Subgraph,
    TensorBox,
)
from ..lowering import (
    _full,
    check_and_broadcast_indices,
    empty,
    empty_like,
    empty_strided,
    expand,
    index_output_size_and_inner_fn,
    lowerings,
    register_lowering,
    to_dtype,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    realize_inputs,
    SymbolicGridFn,
    TritonTemplate,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten
Expr = sympy.Expr


def construct_strides(
    sizes: Sequence[int],
    fill_order: Sequence[int],
) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    assert len(sizes) == len(fill_order), (
        "Length of sizes must match the length of the fill order"
    )
    strides = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def infer_dense_strides(size: Sequence[int], orig_strides: Sequence[int]):
    """This is a mirror of the same function in aten/src/ATen/ExpandUtils.cpp

    Args:
        size: The size of the output tensor
        orig_strides: The strides of the input tensor
    Returns:
        List[int]: Dense non-overlapping strides that preserve the input tensor's layout permutation.
        The returned strides follow the same stride propagation rules as TensorIterator. This matches
        The behavior of empty_like()
    """
    fill_order = get_fill_order(orig_strides, V.graph.sizevars.shape_env)
    return construct_strides(size, fill_order)


@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, ceil_div(n_queries, query_block_size), 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    return (cdiv(num_queries, meta["BLOCK_M"]), batch_size * q_heads, 1)


def create_placeholder(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    size: Optional[list[int]] = None,
) -> TensorBox:
    """Creates a placeholder input buffers for producing subgraph_output."""
    input_buffer = InputBuffer(
        name=name,
        layout=FixedLayout(
            device,
            dtype,
            size if size else [],
            FlexibleLayout.contiguous_strides(size) if size else [],
        ),
    )
    return TensorBox.create(input_buffer)


def maybe_realize(args: list[Optional[IRNode]]):
    """Accepts a list of optional IRNodes and returns a list of realized IRNodes"""
    return tree_map(
        lambda x: (
            realize_inputs(x)
            if x is not None and not isinstance(x, sympy.Symbol)
            else x
        ),
        args,
    )


def get_float32_precision():
    if (
        torch.get_float32_matmul_precision() == "highest"
        or torch.version.hip
        or torch.mtia.is_available()
    ):
        return "'ieee'"
    else:
        return "'tf32'"


def zeros_and_scatter_lowering(shape: list[int], indices, values):
    # Always accumulate into fp32 then cast
    grad = _full(0, values.get_device(), torch.float32, shape)
    assert isinstance(grad, TensorBox)
    grad.realize()
    x_size = grad.get_size()
    values = to_dtype(values, grad.get_dtype())
    indices_loaders = [i.make_loader() if i is not None else None for i in indices]
    indices, tensor_indices = check_and_broadcast_indices(indices, grad.get_device())
    # We can use the first one since they are all required to be the same size
    tensor_size = list(indices[tensor_indices[0]].get_size())
    indexed_size = [x_size[i] for i in range(len(indices))]

    expected_vals_size, inner_fn = index_output_size_and_inner_fn(
        x_size,
        indices,
        tensor_indices,
        tensor_size,
        indices_loaders,
        indexed_size,
        None,
        check=True,
    )

    values = expand(values, expected_vals_size)
    device = grad.get_device()
    assert device is not None
    scatter = Scatter(
        device=device,
        dtype=grad.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=inner_fn,
        scatter_mode="atomic_add",
    )

    buffer = ComputedBuffer(
        name=grad.data.data.name,  # type: ignore[attr-defined]
        layout=MutationLayoutSHOULDREMOVE(grad),
        data=scatter,
    )
    return buffer


SubgraphResults = Union[list[Optional[ComputedBuffer]], Optional[ComputedBuffer]]


def build_subgraph_module_buffer(
    args: list[TensorBox], graph_module: torch.fx.GraphModule
) -> SubgraphResults:
    """This function's goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph. Contains both fixed and lifted inputs.
        subgraph: The Subgraph ir for which to produce the output node
    """
    from ..subgraph_lowering import PointwiseSubgraphLowering

    pw_subgraph = PointwiseSubgraphLowering(
        graph_module,
        root_graph_lowering=V.graph,
        allowed_mutations=OrderedSet([torch.ops.flex_lib.zeros_and_scatter.default]),
        additional_lowerings={
            torch.ops.flex_lib.zeros_and_scatter.default: zeros_and_scatter_lowering
        },
    )
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*args)

    # Since we are allowing mutations/buffer creation, we need to register any fresh buffers
    # creating during the pointwise subgraph lowering
    if len(pw_subgraph.buffers) > 0:
        for buffer in pw_subgraph.buffers:
            V.graph.register_buffer(buffer)

    def convert_output_node_to_buffer(output_buffer) -> Optional[ComputedBuffer]:
        if output_buffer is None:
            return None
        if isinstance(output_buffer, ComputedBuffer):
            # These nodes are coming from the output of zeros_and_scatter
            return output_buffer
        assert isinstance(output_buffer, TensorBox), (
            "The output node for flex attention's subgraph must be a TensorBox, but got: ",
            type(output_buffer),
        )
        assert isinstance(output_buffer.data, StorageBox), (
            "The output node for the flex attention subgraph must be a StorageBox, but got: ",
            type(output_buffer),
        )
        subgraph_buffer = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=output_buffer.data.get_device(),
                dtype=output_buffer.data.get_dtype(),
                size=output_buffer.data.get_size(),
            ),
            data=output_buffer.data.data,  # type: ignore[arg-type]
        )
        return subgraph_buffer

    return tree_map(convert_output_node_to_buffer, pw_subgraph.graph_outputs)


def build_subgraph_buffer(args: list[TensorBox], subgraph: Subgraph) -> SubgraphResults:
    return build_subgraph_module_buffer(args, subgraph.graph_module)


# Inner Triton functions shared by flex_attention & split-k decoding kernels.
compute_next_offset_func = r"""
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset
"""

get_bounded_indices_func = r"""
@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices
"""


load_checked_block = r"""
@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
"""

load_checked_2d = r"""
@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_DIM: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_DIM), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)
"""

compute_flex_attention = r"""
{{def_kernel("Q", "K", "V", "LSE", "KV_NUM_BLKS", "KV_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX")}}
    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    #
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad
    #
    # (Modifiable) Performance tuning options
    # BLOCK_M: The thread block size across the seqlen dim of Q.
    # BLOCK_N: Iterate over BLOCK_N across the seqlen dim of K/V in each thread block.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # BLOCKS_ARE_CONTIGUOUS: Is it guaranteed that all blocks in the mask are
    # contiguous? If so, we don't need to do an indirect jump for every block

    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qk = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kk = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vk = {{stride("V")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0)
    off_zq = tl.program_id(1) // HQ
    off_hq = tl.program_id(1) % HQ

    # We support two cases for batch dimension. a) (ZKV == ZQ) where off_zkv = off_zq.
    # b) (ZKV == 1 and ZQ > 1) where KV is broadcasted along the batch dimension and off_zkv=0.
    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS
    off_g = off_hq % GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    sparse_idx_z = off_zq % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
    stride_kv_idx_h = {{stride("KV_IDX", 1)}}
    stride_kv_idx_m = {{stride("KV_IDX", 2)}}

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # KV_IDX and KV_NUM_BLKS are always contiguous.
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m  # noqa: B950

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Q_LEN, QK_HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_start * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
        order=(1, 0)
    )
    q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We don't know anything "special" about these blocks, so we need to apply
    # both score_mod and mask_mod to it
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(QK_HEAD_DIM, KV_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, kv_start),
        block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(KV_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
        order=(1, 0)
    )
    offs_n = kv_start + tl.arange(0, BLOCK_N)

    acc, l_i, m_i = forward_inner(
        {{gen_argdefs()}},
        q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_zq, off_hq, offs_m[:, None], offs_n[None, :],
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )

    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
        kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(QK_HEAD_DIM, KV_LEN),
            strides=(stride_kk, stride_kn),
            offsets=(0, kv_start),
            block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner(
            {{gen_argdefs()}},
            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )


    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1, l_i)

    acc = acc / l_i[:, None]
    idx_zq = tl.program_id(1) // HQ
    idx_hq = tl.program_id(1) % HQ
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :]

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

    {{store_output(("idx_zq", "idx_hq", "idx_m", "idx_d"), "acc", "mask")}}

    if OUTPUT_LOGSUMEXP:
        off_hz = tl.program_id(1)
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)
 """


compute_forward_inner = r"""
@triton.jit
def forward_inner(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    {{gen_defines() | indent_except_first(1)}}

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                {{gen_argdefs()}},
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                {{gen_argdefs()}},
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )

        # update pointers
        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, offset))

        offs_n = offs_n + offset

    return acc, l_i, m_i

"""


compute_forward_block_mn = r"""
@triton.jit
def forward_block_mn(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    {{gen_defines() | indent_except_first(1)}}

    # -- load k --
    # NB reversed order to since K is transposed
    k = load_checked_block(K_block_ptr, SAFE_HEAD_DIM, IS_DIVISIBLE)
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
    # which is larger than the actual number of elements. To avoid access memory out of bound,
    # we need to mask out the elements that are out of Q_LEN & KV_LEN.
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qk",
        b="off_z",
        h="off_h",
        m="m",
        n="n",
        out="qk"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        {{ modification(
            subgraph_number=1,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
        ) | indent_except_first(2) }}

        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    v = load_checked_block(V_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

"""


flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=compute_flex_attention
    + compute_forward_inner
    + compute_next_offset_func
    + compute_forward_block_mn
    + load_checked_block
    + get_bounded_indices_func,
)


def _use_flex_decoding(query, kv_indices, kernel_options, enable_gqa):
    """Decide which kernel to use, return true if use flex decoding kernel.
    Note:
       Since the number of splits is calculated based of the the number of batch and head dims
       we need to ensure that the batch and head dims are statically known. Otherwise we just
       use the main flex_attention kernel.
    """
    force_flex = kernel_options.get("FORCE_USE_FLEX_ATTENTION", False)
    short_query_length = V.graph.sizevars.evaluate_expr(
        sympy.Lt(query.get_size()[-2], 128)
    )
    non_zero_length = V.graph.sizevars.evaluate_expr(sympy.Gt(query.get_size()[-2], 0))
    static_batch = isinstance(query.get_size()[0], (int, sympy.Integer))
    static_num_heads = isinstance(query.get_size()[1], (int, sympy.Integer))
    if enable_gqa:
        # in the current flex decoding triton kernel, grouped query heads for the
        # same kv head are handled by the same block. So it's hard to support different
        # kv num blocks for grouped query heads. We just fall back to main flex_attention
        # kernel where each query head is handled by a separate block.
        valid_block_mask_num_heads = V.graph.sizevars.evaluate_expr(
            sympy.Eq(kv_indices.get_size()[1], 1)
        )
    else:
        valid_block_mask_num_heads = V.graph.sizevars.evaluate_expr(
            sympy.Or(
                sympy.Eq(kv_indices.get_size()[1], 1),
                sympy.Eq(kv_indices.get_size()[1], query.get_size()[1]),
            )
        )
    return (
        not force_flex
        and short_query_length
        and static_batch
        and static_num_heads
        and non_zero_length
        and valid_block_mask_num_heads
    )


_h100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (32, 64, 4, 3),
    (torch.float32, 256): (32, 32, 4, 3),
    (torch.bfloat16, 64): (128, 128, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (64, 32, 4, 3),
    (torch.float16, 64): (128, 128, 4, 3),
    (torch.float16, 128): (128, 128, 8, 3),
    (torch.float16, 256): (64, 32, 4, 3),
}

_a100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (128, 32, 4, 3),
    (torch.float32, 256): (64, 16, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (32, 64, 4, 3),
    (torch.float16, 64): (128, 64, 4, 3),
    (torch.float16, 128): (128, 64, 8, 3),
    (torch.float16, 256): (32, 64, 4, 3),
}

_rocm_default_config = {
    (torch.float32, 64): (128, 32, 4, 1),
    (torch.float32, 128): (128, 32, 4, 1),
    (torch.float32, 256): (64, 16, 4, 1),
    (torch.bfloat16, 64): (128, 64, 8, 1),
    (torch.bfloat16, 128): (128, 64, 8, 1),
    (torch.bfloat16, 256): (32, 64, 8, 1),
    (torch.float16, 64): (128, 64, 8, 1),
    (torch.float16, 128): (128, 64, 8, 1),
    (torch.float16, 256): (32, 64, 4, 1),
}


class Mode(Enum):
    fwd = auto()
    bwd = auto()


def _get_rocm_config(query, mode: Mode) -> tuple[int, int, int, int]:
    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.evaluate_static_shape(query.get_size()[-1])
    fwd_config = None

    if mode == Mode.fwd:
        if head_dim <= 256:
            if dtype == torch.float32:
                fwd_config = (64, 64, 4, 1)
            else:
                fwd_config = (128, 64, 8, 1)
            fwd_config = _rocm_default_config.get((dtype, head_dim), fwd_config)
        else:  # modest hardware or extremely large head_dim
            if dtype == torch.float32:
                fwd_config = (32, 16, 4, 1)
            else:
                fwd_config = (64, 32, 4, 1)
        return fwd_config
    else:  # bwd
        assert mode == Mode.bwd
        if dtype == torch.float32:
            return (16, 16, 4, 1)
        elif head_dim <= 256:
            if head_dim == 64:
                return (64, 64, 4, 1)
            elif head_dim == 128:
                return (64, 128, 8, 1)
            else:
                return (64, 64, 4, 1)
        else:  # modest hardware or extremely large head_dim
            return (16, 16, 4, 1)


def _get_nv_config(query, mode: Mode) -> tuple[int, int, int, int]:
    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.evaluate_static_shape(query.get_size()[-1])
    fwd_config = None
    bwd_config = None
    capability = torch.cuda.get_device_capability()

    if mode == Mode.fwd:
        if head_dim <= 256:
            if dtype == torch.float32:
                fwd_config = (64, 64, 4, 3)
            else:
                fwd_config = (128, 64, 4, 3)
            if capability >= (9, 0):
                fwd_config = _h100_default_config.get((dtype, head_dim), fwd_config)
            elif capability >= (8, 0):
                fwd_config = _a100_default_config.get((dtype, head_dim), fwd_config)
        else:  # modest hardware or extremely large head_dim
            if dtype == torch.float32:
                fwd_config = (32, 16, 4, 3)
            else:
                fwd_config = (64, 32, 4, 3)
        return fwd_config

    else:  # bwd
        assert mode == Mode.bwd
        if dtype == torch.float32:
            bwd_config = (16, 16, 4, 1)
        elif head_dim <= 256 and capability >= (9, 0):  # H100
            if head_dim == 64:
                bwd_config = (64, 64, 4, 3)
            elif head_dim == 128:
                bwd_config = (64, 128, 8, 3)
            else:
                bwd_config = (64, 64, 4, 2)
        elif capability >= (8, 0):
            if head_dim >= 64:
                bwd_config = (32, 128, 4, 3)
            elif head_dim == 128:
                # SM86/89 have smaller shared memory sizes
                num_stages = 3 if capability[-1] == 0 else 2
                bwd_config = (64, 64, 4, num_stages)
            else:
                bwd_config = (64, 64, 4, 2)
        else:  # modest hardware or extremely large head_dim
            bwd_config = (16, 16, 4, 1)
        return bwd_config


def _get_default_config_fwd(query) -> tuple[int, int, int, int]:
    if torch.version.hip is None:
        return _get_nv_config(query, mode=Mode.fwd)
    else:
        return _get_rocm_config(query, mode=Mode.fwd)


def _get_default_config_bwd(query) -> tuple[int, int, int, int]:
    if torch.version.hip is None:
        return _get_nv_config(query, mode=Mode.bwd)
    else:
        return _get_rocm_config(query, mode=Mode.bwd)


def create_num_blocks_fake_generator(sparse_indices):
    # The idea here is that we need to create a real tensor with real data
    # that's representative for benchmarking.
    # For example, returning all zeros for the `kv_num_blocks` input would mean
    # that we are computing 0 blocks for each row, which would provide bogus
    # autotuning results.
    #
    # In this case, we choose to use min(16, max_block) blocks, because I
    # (Horace) think it'll probably result in pretty representative performance.
    # If it's too short then prefetching won't help. If it's too long then
    # autotuning will take longer for no good reason.
    def create_num_blocks_fake(x) -> torch.Tensor:
        num_blocks_for_autotuning = V.graph.sizevars.size_hint(sparse_indices.shape[-1])
        size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
        return torch.full(
            size,
            num_blocks_for_autotuning,
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_num_blocks_fake


def create_indices_fake(x) -> torch.Tensor:
    size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
    indices = torch.arange(0, size[-1], dtype=x.get_dtype(), device=x.get_device())
    indices = indices.expand(size).contiguous()
    return indices


from torch._inductor.kernel.flex_decoding import create_flex_decoding_kernel

from ..codegen.cpp_flex_attention_template import CppFlexAttentionTemplate


def check_cpu_supported():
    import os
    import sys

    requires_avx2_on_cpu = (
        torch.cpu._is_avx2_supported() and os.getenv("ATEN_CPU_CAPABILITY") != "default"
    )
    supported = (
        requires_avx2_on_cpu
        and not torch.xpu.is_available()
        and not sys.platform == "darwin"
    )
    return supported


def lower_cpu(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    if kernel_options["OUTPUT_LOGSUMEXP"]:
        raise NotImplementedError(
            "torch.compile on CPU only supports inference and `return_lse` is not supported yet."
        )
    if not check_cpu_supported():
        raise NotImplementedError(
            "torch.compile on current platform is not supported for CPU."
        )

    fake_buffers: list[Buffer] = []  # noqa: F821

    # [Note] Handle the case where the split sizes are not statically known.
    # The value of cur_qSplitSize and cur_kvSplitSize are decided during runtime.
    # We use symbols to represent them during the compilation here.
    # They'll be replaced by the string "cur_qSplitSize" and "cur_kvSplitSize" in
    # the modification function of the CppFlexAttentionTemplate class.
    cur_qSplitSize = V.graph.sizevars.shape_env.create_unbacked_symint().node.expr
    cur_kvSplitSize = V.graph.sizevars.shape_env.create_unbacked_symint().node.expr
    shape_env = V.graph.sizevars.shape_env

    # We don't know the concret value of cur_qSplitSize and cur_kvSplitSize during the compilation.
    # Mark symbols > 1 to ensure broadcasting is always applied.
    # This avoids treating them as equal when `eq(var, 1)` is evaluated in `broadcast_symbolic_shapes`.
    shape_env.var_to_range[cur_qSplitSize] = ValueRanges(2, int_oo)
    shape_env.var_to_range[cur_kvSplitSize] = ValueRanges(2, int_oo)

    score_dtype = torch.float
    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device(), size)
        for name, dtype, size in [
            ("score", score_dtype, [cur_qSplitSize, cur_kvSplitSize]),
            ("b", torch.int64, []),
            ("h", torch.int64, []),
            ("q_idx", torch.int64, [cur_qSplitSize, 1]),
            ("kv_idx", torch.int64, [1, cur_kvSplitSize]),
        ]
    ]
    subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), subgraph
    )
    if subgraph_buffer is not None:
        if isinstance(subgraph_buffer, list):
            for _buf in subgraph_buffer:
                if _buf is not None:
                    _buf.freeze_layout()
        else:
            subgraph_buffer.freeze_layout()
    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device(), size)
        for name, dtype, size in [
            ("score", score_dtype, [cur_qSplitSize, cur_kvSplitSize]),
            ("b", torch.int64, []),
            ("h", torch.int64, []),
            ("q_idx", torch.int64, [cur_qSplitSize, 1]),
            ("kv_idx", torch.int64, [1, cur_kvSplitSize]),
        ]
    ]

    # The original mask_graph works on a scalar and only includes
    # the logic of calculating the mask value.
    # We need to add the logic of applying the mark to the qk_data tensor
    # into the graph for the later codegen of this part.
    # Example:
    #   mask_graph:
    #   def mask_fn(b, h, q_idx, kv_idx):
    #       mask = q_idx >= kv_idx
    #       return mask
    #   The converted_mask_graph should be:
    #   def converted_mask_fn(qk_data, b, h, q_idx, kv_idx):
    #       mask = q_idx >= kv_idx
    #       qk_data = torch.where(mask, qk_data, torch.full_like(qk_data, -float("inf")))
    #       return qk_data
    def convert_mask_graph_module(mask_graph):
        gm = copy.deepcopy(mask_graph.graph_module)
        graph = gm.graph
        # Add qk_data as the first input
        with graph.inserting_before(next(iter(graph.nodes))):
            qk_data_node = graph.placeholder("qk_data")

        # Find the node that returns the mask
        output_node = None
        for node in graph.nodes:
            if node.op == "output":
                output_node = node
                break

        # Get the mask node
        assert output_node is not None
        mask_node = output_node.args[0]

        size_node = [cur_qSplitSize, cur_kvSplitSize]
        # Create a new node for torch.full
        with graph.inserting_after(mask_node):
            full_node = graph.call_function(
                torch.full,
                args=(size_node, -float("inf")),
                kwargs={"dtype": score_dtype},
            )

        # Create a new node for torch.where
        with graph.inserting_after(full_node):
            where_node = graph.call_function(
                torch.ops.aten.where, args=(mask_node, qk_data_node, full_node)
            )

        # Update the output node to return the result of torch.where
        output_node.args = (where_node,)

        graph.lint()
        converted = torch.fx.GraphModule(gm, graph)
        return converted

    converted_mask_graph_module = convert_mask_graph_module(mask_graph)

    mask_graph_buffer = build_subgraph_module_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers),
        converted_mask_graph_module,
    )

    # Clear the pending fresh unbacked symbols that are created for cur_qSplitSize and cur_kvSplitSize in the current kernel.
    pending = V.graph.sizevars.shape_env.pending_fresh_unbacked_symbols
    V.graph.sizevars.shape_env.pending_fresh_unbacked_symbols = [
        x for x in pending if x not in (cur_qSplitSize, cur_kvSplitSize)
    ]

    buffer_list = (
        placeholder_inps
        + list(score_mod_other_buffers)
        + mask_graph_placeholder_inps
        + list(mask_mod_other_buffers)
    )
    for item in buffer_list:
        if isinstance(item, TensorBox):
            fake_buffers.append(item.data.data)  # type: ignore[attr-defined]

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
    )

    if len(OrderedSet([query.get_name(), key.get_name(), value.get_name()])) != 3:
        raise NotImplementedError(
            "Unsupported for now if query, key, value are the same buffer."
        )
    if query.get_dtype() not in [torch.float, torch.bfloat16, torch.float16]:
        raise NotImplementedError(
            "`torch.float` , `torch.float16` and `torch.bfloat16` are supported in FlexAttention for CPU device. "
            f"Found input tensors are `{query.get_dtype()}`."
        )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)
    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    B = Bq

    # Construct output layout with strides matching the query.
    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, query.get_stride())

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, Hq, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in out_strides],
    )
    _choices: list[Any] = []
    input_nodes = [query, key, value, kv_num_blocks, kv_indices]
    if not full_kv_num_blocks:
        no_full_kv_block = True
    else:
        no_full_kv_block = False
        input_nodes += [full_kv_num_blocks]
    has_other_buffer = False
    kernel_input_name_to_buffer = {}
    if score_mod_other_buffers or mask_mod_other_buffers:
        has_other_buffer = True

        for prefix, buffers in [
            ("score_others", score_mod_other_buffers),
            ("mask_others", mask_mod_other_buffers),
        ]:
            kernel_input_name_to_buffer.update(
                {f"{prefix}_{i}": buf for i, buf in enumerate(buffers)}
            )
        input_nodes += [
            value
            for value in kernel_input_name_to_buffer.values()
            if not isinstance(value, sympy.Symbol)
        ]

    skip_mask_score = kernel_options.get("SKIP_MASK_SCORE", False)
    # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_Q_BLOCK_SIZE)
    assert V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_q, sympy.Mul(kv_indices.get_size()[-2], SPARSE_Q_BLOCK_SIZE))
    ), (
        "Q seqlen must be smaller than the block_mask size in the Q dimension, considering pass a larger block_mask."
    )
    assert V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_kv, sympy.Mul(kv_indices.get_size()[-1], SPARSE_KV_BLOCK_SIZE))
    ), (
        "KV seqlen must be smaller than the block_mask size in the KV dimension, considering pass a larger block_mask."
    )
    CppFlexAttentionTemplate.add_choices(
        choices=_choices,
        input_nodes=input_nodes,
        layout=layout,
        scale=scale,
        score_mod=None if skip_mask_score else subgraph_buffer,
        mask_mod=None if skip_mask_score else mask_graph_buffer,
        kv_block_size=SPARSE_KV_BLOCK_SIZE,
        has_other_buffer=has_other_buffer,
        no_full_kv_block=no_full_kv_block,
        fake_buffers=fake_buffers,
        len_score_other=len(score_mod_other_buffers),
        len_mask_other=len(mask_mod_other_buffers),
        kernel_input_name_to_buffer=kernel_input_name_to_buffer,
        block_vars=(cur_qSplitSize, cur_kvSplitSize),
    )
    inputs_for_autotuning = [
        query,
        key,
        value,
    ]
    res = autotune_select_algorithm(
        "flex_attention",
        _choices,
        inputs_for_autotuning,
        layout,
    )
    return (res,)


def is_power_of_2(n):
    return n != 0 and ((n & (n - 1)) == 0)


def next_power_of_two(n):
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


def set_head_dim_values(
    kernel_options: dict[str, Any], qk_head_dim, v_head_dim, graph_sizevars
):
    """
    Mutates kernel options, adding head dimension calculations.

    Args:
        kernel_options: Dictionary to populate with options
        qk_head_dim: Query/Key head dimension
        v_head_dim: Value head dimension
        graph_sizevars: Graph size variables object with evaluate_static_shape method

    """
    # QK dimensions
    qk_head_dim_static = graph_sizevars.evaluate_static_shape(qk_head_dim)
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim_static)
    kernel_options.setdefault(
        "QK_HEAD_DIM_ROUNDED", next_power_of_two(qk_head_dim_static)
    )

    # V dimensions
    v_head_dim_static = graph_sizevars.evaluate_static_shape(v_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim_static)
    kernel_options.setdefault(
        "V_HEAD_DIM_ROUNDED", next_power_of_two(v_head_dim_static)
    )

    # Safety flag
    kernel_options.setdefault(
        "SAFE_HEAD_DIM",
        is_power_of_2(qk_head_dim_static) and is_power_of_2(v_head_dim_static),
    )


# TODO: We probably also need a layout constraint?
@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options,
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    if query.get_device().type == "cpu":
        return lower_cpu(
            query,
            key,
            value,
            subgraph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

    # below is cuda path if device is not cpu
    # tl.dot does not support embedding size less than 16
    small_dqk = V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-1], 16))
    small_dv = V.graph.sizevars.evaluate_expr(sympy.Lt(value.get_size()[-1], 16))
    if small_dqk or small_dv:
        raise NotImplementedError(
            f"NYI: embedding dimension of the query, key, and value must be "
            f"at least 16 but got E={query.get_size()[-1]} and Ev={value.get_size()[-1]}"
        )

    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), subgraph
    )

    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )

    kernel_options = dict(kernel_options)
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.evaluate_static_shape(v)
        if isinstance(v, sympy.Symbol)
        else v
        for k, v in kernel_options.items()
    }
    kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
    enable_gqa = V.graph.sizevars.evaluate_expr(
        sympy.Ne(query.get_size()[1], key.get_size()[1]),
    )
    if _use_flex_decoding(query, kv_indices, kernel_options, enable_gqa):
        return create_flex_decoding_kernel(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
    )

    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    assert V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)), (
        f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
    )
    assert V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_q, 0)), (
        "Query length must be greater than 0"
    )
    assert V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_kv, 0)), (
        "Key length must be greater than 0"
    )

    B = Bq

    if seq_len_q % 128 != 0 or seq_len_kv % 128 != 0:
        kernel_options.setdefault("IS_DIVISIBLE", False)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", True)

    # Reuse query strides for output layout despite different last dimension.
    # This works because only the last dim differs and we check it is contiguous.
    q_strides = query.get_stride()
    assert q_strides[-1] == 1, "Query must be contiguous in the last dimension"

    # Construct output layout with strides matching the query.
    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, Hq, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in out_strides],
    )
    # see NOTE:[TritonTemplates with multiple outputs]
    logsumexp_shape = [B, Hq, seq_len_q]
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=torch.float32,  # The logsumexp is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    kernel_options.setdefault("SM_SCALE", scale)

    # Determine GQA broadcast factor.
    gqa_shared_heads = Hq // Hkv
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Inside of Triton kernel, only apply partial masking if partial blocks are computed.
    # full_kv_num_blocks is None if partial blocks are not computed
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    choices: list[Any] = []
    configs: list[tuple[int, int, int, int]] = []
    configs.append(_get_default_config_fwd(query))
    if config.max_autotune:
        configs += [
            (128, 64, 4, 3),
            (128, 128, 4, 3),
            (128, 128, 8, 2),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
        ]

        # On ROCm convert num_stages to 1 to avoid shmem issues
        if torch.version.hip:
            configs = [(c[0], c[1], c[2], 1) for c in configs]

    # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_Q_BLOCK_SIZE)

    # ROCm specific considerations
    if torch.version.hip:
        kernel_options["kpack"] = 2

    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    original_kernel_options = kernel_options.copy()
    for BLOCK_M, BLOCK_N, num_warps, num_stages in configs:
        if SPARSE_KV_BLOCK_SIZE % BLOCK_N != 0 or SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0:
            if len(configs) == 1:
                raise ValueError(
                    f"Q and KV block size must be divisible by BLOCK_M and BLOCK_N. We "
                    f"got Q_BLOCK_SIZE={SPARSE_Q_BLOCK_SIZE} and KV_BLOCK_SIZE={SPARSE_KV_BLOCK_SIZE}."
                )
            continue

        cur_kernel_options = original_kernel_options.copy()
        # Performance tuning
        # Triton parameters
        # Remove prefix for forward kernels options and delete backward kernel options.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault("num_stages", num_stages)
        cur_kernel_options.setdefault("num_warps", num_warps)
        cur_kernel_options.setdefault("BLOCK_M", BLOCK_M)
        cur_kernel_options.setdefault("BLOCK_N", BLOCK_N)
        # Blocksparse options
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

        error = flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout,
            subgraphs=[
                subgraph_buffer,
                mask_graph_buffer,
            ],
            mutated_inputs=[
                logsumexp,
            ],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )
        if error is not None and len(configs) == 1:
            raise error
    inputs_for_autotuning = (
        [
            query,
            key,
            value,
            logsumexp,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )
    input_gen_fns = {
        4: create_num_blocks_fake_generator(kv_indices),
        5: create_indices_fake,
        6: create_num_blocks_fake_generator(full_kv_indices),
        7: create_indices_fake,
    }
    return (
        autotune_select_algorithm(
            "flex_attention",
            choices,
            inputs_for_autotuning,
            layout,
            input_gen_fns=input_gen_fns,
        ),
        logsumexp,
    )


# ---------------------------- Backward HOP Implementation ----------------------------


def flex_attention_backward_grid(
    batch_size, q_heads, num_queries, d_model, kv_heads, num_key_value, meta
):
    """How is this kernel parallelized?
    Currently this is only parallelizing over batch* kv_heads, but we can, and want to
    parallelize over ceil_div(q_heads//kv_heads * num_key_value, key_value_block_size).
    To do this will either require atomic updates to some grad values or to have a two pass kernel design.
    """
    import triton

    return (
        triton.cdiv(num_queries, meta["BLOCK_M2"]) * (q_heads // kv_heads)
        + triton.cdiv(num_key_value, meta["BLOCK_N1"]),
        1,
        batch_size * kv_heads,
    )


flex_attention_backward_template = TritonTemplate(
    name="flex_attention_backward",
    grid=flex_attention_backward_grid,
    source=r"""
{{def_kernel("Q", "K", "V", "LSE", "DELTA", "DO", "DQ", "DV", "KV_NUM_BLKS", "KV_IDX", "Q_NUM_BLKS", "Q_IDX", "FULL_KV_NUM_BLKS", "FULL_KV_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}
    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT*DO, axis=-1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key, is the written to via the store_output call due to some limitations with
    # inductor codegen
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    # (Modifiable) Performance tuning options
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # Q_NUM_BLKS: The number of Q blocks (that may or may not require masking) for each query.
    # Q_IDX: The indices of Q blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_Q_NUM_BLKS: The number of fully unmasked Q blocks (so we don't need masking) for each query.
    # FULL_Q_IDX: The indices of fully unmasked Q blocks (so we don't need masking) for each query.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qd = {{stride("Q")}}
    stride_kz, stride_kh, stride_kn, stride_kd = {{stride("K")}}
    stride_vz, stride_vh, stride_vn, stride_vd = {{stride("V")}}
    stride_doz, stride_doh, stride_dom, stride_dod = {{stride("DO")}}

    stride_dqz, stride_dqh, stride_dqm, stride_dqd = {{stride("DQ")}}
    stride_dvz, stride_dvh, stride_dvm, stride_dvd = {{stride("DV")}}

    ZQ = {{size("Q", 0)}}
    HQ = {{size("Q", 1)}}
    HKV = {{size("K", 1)}}
    Q_LEN = {{size("Q", 2)}}
    ZKV = {{size("K", 0)}}
    KV_LEN = {{size("K", 2)}}

    MATMUL_PRECISION = Q.dtype.element_ty

    pid = tl.program_id(0)
    NUM_KV_BLOCKS = tl.cdiv(KV_LEN, BLOCK_N1)
    NUM_Q_BLOCKS = tl.cdiv(Q_LEN, BLOCK_M2)

    off_hz = tl.program_id(2)
    off_zq = off_hz // HKV # q batch idx
    off_hkv = off_hz % HKV # kv head idx
    off_zkv = off_zq % ZKV # kv batch idx

    SPARSE_Z = {{size("KV_NUM_BLKS", 0)}}
    SPARSE_HQ = {{size("KV_NUM_BLKS", 1)}}

    sparse_idx_z = off_zq % SPARSE_Z

    k_adj = (stride_kh * off_hkv + stride_kz * off_zkv).to(tl.int64)
    v_adj = (stride_vh * off_hkv + stride_vz * off_zkv).to(tl.int64)
    # first compute broadcasted dv of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
    # then reduce to dv of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
    dv_adj = (stride_dvh * off_hkv + stride_dvz * off_zq).to(tl.int64)

    # offset K, V, DV pointers for batch/kv-head
    K += k_adj
    V += v_adj
    DV += dv_adj

    RCP_LN2 = 1.44269504
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    if pid >= NUM_KV_BLOCKS:
        off_pid = pid - NUM_KV_BLOCKS
        # THIS BLOCK DOES DQ
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M2)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
        off_hq2 = off_pid // NUM_Q_BLOCKS + off_hkv * GQA_SHARED_HEADS
        start_m2_block = off_pid % NUM_Q_BLOCKS
        off_pid_mask = start_m2_block // SPARSE_Q_MULTIPLE
        stride_kv_num_blks_h = {{stride("KV_NUM_BLKS", 1)}}
        stride_kv_idx_h = {{stride("KV_IDX", 1)}}
        stride_kv_idx_m = {{stride("KV_IDX", 2)}}

        sparse_idx_hq2 = off_hq2 % SPARSE_HQ
        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq2

        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + off_pid_mask
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + off_pid_mask * stride_kv_idx_m  # noqa: B950

        # Offset Q, DQ, DO, DELTA & LSE. These inputs are offseted by query heads.
        q_adj2 = (stride_qh * off_hq2 + stride_qz * off_zq).to(tl.int64)
        do_adj2 = (stride_doh * off_hq2 + stride_doz * off_zq).to(tl.int64)
        dq_adj2 = (stride_dqh * off_hq2 + stride_dqz * off_zq).to(tl.int64)
        off_chz2 = ((off_zq * HQ + off_hq2) * Q_LEN).to(tl.int64)

        Q2 = Q + q_adj2
        DO2 = DO + do_adj2
        # TODO: This does not work if DQ is not the same layout as Q (for example,
        # if Q is broadcasted)
        DQ2 = DQ + dq_adj2
        LSE2 = LSE + off_chz2
        DELTA2 = DELTA + off_chz2

        # dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM], dtype=tl.float32)
        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_m2 = start_m2_block * BLOCK_M2
        offs_m2 = start_m2 + tl.arange(0, BLOCK_M2)

        # load Q and do: they stay in SRAM throughout the inner loop.
        q = load_checked_2d(Q2, offs_m2, offs_k, stride_qm, stride_qd, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, QK_HEAD_DIM)
        do = load_checked_2d(DO2, offs_m2, offs_v, stride_dom, stride_dod, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM)

        if PRESCALE_QK:
            q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        if IS_DIVISIBLE:
            Di = tl.load(DELTA2 + offs_m2)
            lse = tl.load(LSE2 + offs_m2)
        else:
            Di = tl.load(DELTA2 + offs_m2, mask=offs_m2 < Q_LEN)
            lse = tl.load(LSE2 + offs_m2, mask=offs_m2 < Q_LEN)
        lse = tl.where(lse == -float("inf"), 0.0, lse)
        lse = lse[:, None]

        # ~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # KV_IDX and KV_NUM_BLKS are always contiguous.
        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        sparse_kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
        dq = bwd_dq_inner(
            {{gen_argdefs()}},
            K, V,
            dq, q, do, Di, lse,
            off_zq, off_hq2, offs_m2, offs_n2,
            stride_kn, stride_kd, stride_vn, stride_vd,
            kv_indices, sparse_kv_num_blocks,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=False,
        )

        if HAS_FULL_BLOCKS:
            # ~~~~~~~~~~~ partial unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
            sparse_kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)

            offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
            dq = bwd_dq_inner(
                {{gen_argdefs()}},
                K, V,
                dq, q, do, Di, lse,
                off_zq, off_hq2, offs_m2, offs_n2,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=True,
            )

        # Write back dQ.
        dq_ptrs = DQ2 + offs_m2[:, None] * stride_dqm + offs_k[None, :] * stride_dqd
        dq *= SM_SCALE
        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=(offs_m2[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM))
    else:
        # THIS BLOCK DOES DK & DV
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

        pid_mask = pid // SPARSE_KV_MULTIPLE

        stride_q_num_blks_h = {{stride("Q_NUM_BLKS", 1)}}
        stride_q_idx_h = {{stride("Q_IDX", 1)}}
        stride_q_idx_n = {{stride("Q_IDX", 2)}}


        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM_ROUNDED], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_n1 = pid * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)

        # load K and V: they stay in SRAM throughout the inner loop.
        k = load_checked_2d(K, offs_n1, offs_k, stride_kn, stride_kd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)
        v = load_checked_2d(V, offs_n1, offs_v, stride_vn, stride_vd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)

        if PRESCALE_QK:
            k = (k * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g

            # Offset Q, DQ, DO, DELTA & LSE. These inputs are offseted by query heads.
            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_zq).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_zq).to(tl.int64)
            dq_adj1 = (stride_dqh * off_hq1 + stride_dqz * off_zq).to(tl.int64)
            off_chz1 = ((off_zq * HQ + off_hq1) * Q_LEN).to(tl.int64)

            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            # TODO: This does not work if DQ is not the same layout as Q (for example,
            # if Q is broadcasted)
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq1

            sparse_q_num_blks_offset = sparse_hz_offset * stride_q_num_blks_h + pid_mask
            sparse_q_idx_offset = sparse_hz_offset * stride_q_idx_h + pid_mask * stride_q_idx_n  # noqa: B950

            # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Q_IDX and Q_NUM_BLKS are always contiguous.
            q_indices = Q_IDX + sparse_q_idx_offset
            q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)

            offs_m1 = q_start + tl.arange(0, BLOCK_M1)
            dk, dv = bwd_dkdv_inner(
                {{gen_argdefs()}},
                Q1, DO1, DELTA1, LSE1,
                dk, dv, k, v,
                off_zq, off_hq1, offs_n1, offs_m1,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=False,
            )


            if HAS_FULL_BLOCKS:
                # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FULL_Q_IDX and FULL_Q_NUM_BLKS are always contiguous.
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)

                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
                dk, dv = bwd_dkdv_inner(
                    {{gen_argdefs()}},
                    Q1, DO1, DELTA1, LSE1,
                    dk, dv, k, v,
                    off_zq, off_hq1, offs_n1, offs_m1,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION,
                    IS_FULL_BLOCKS=True,
                )

        # Write back dV and dK.
        dv_ptrs = DV + offs_n1[:, None] * stride_dvm + offs_v[None, :] * stride_dvd

        index_n = offs_n1[:, None]
        index_k = offs_k[None, :]
        index_v = offs_v[None, :]

        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM))

        dk *= SM_SCALE

        if SAFE_HEAD_DIM:
            mask = index_n < KV_LEN
        else:
            mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)

        # first compute broadcasted dk of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
        # then reduce to dk of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
        {{store_output(("off_zq", "off_hkv", "index_n", "index_k"), "dk", "mask", indent_width=8)}}

@triton.jit
def bwd_dq_inner(
    {{gen_argdefs()}},
    K, V,  # pointers
    dq, q, do, Di, lse,
    off_z, off_hq, offs_m2, offs_n2,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    {{gen_defines() | indent_except_first(1) }}
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = {{size("Q", 2)}}
    KV_LEN = {{size("K", 2)}}

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    kT_ptrs = K + offs_n2[None, :] * stride_kn + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n2[None, :] * stride_vn + offs_v[:, None] * stride_vd
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

    hi = tl.minimum(sparse_kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N2), 1))
    if not IS_DIVISIBLE:
        if hi >= 1:
            for start_n in range(0, hi - 1):
                dq = bwd_dq_block_mn(
                    {{gen_argdefs()}},
                    dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                    off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
                    stride_kn, stride_kd, stride_vn, stride_vd,
                    kv_indices, sparse_kv_num_blocks,
                    MATMUL_PRECISION, RCP_LN2,
                    IS_FULL_BLOCKS,
                )

                # Increment pointers.
                offset = get_offset_for_next_block(
                    start_n, kv_indices, sparse_kv_num_blocks,
                    SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2, BLOCKS_ARE_CONTIGUOUS
                )

                kT_ptrs += offset * stride_kn
                vT_ptrs += offset * stride_vn

                offs_n2 += offset

            dq = bwd_dq_block_mn(
                {{gen_argdefs()}},
                dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )
    else:
        for start_n in range(0, hi):
            dq = bwd_dq_block_mn(
                {{gen_argdefs()}},
                dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )

            # Increment pointers.
            offset = get_offset_for_next_block(
                start_n, kv_indices, sparse_kv_num_blocks,
                SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2, BLOCKS_ARE_CONTIGUOUS
            )

            kT_ptrs += offset * stride_kn
            vT_ptrs += offset * stride_vn

            offs_n2 += offset

    return dq


@triton.jit
def bwd_dq_block_mn(
    {{gen_argdefs()}},
    dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
    off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1)}}

    # NB reversed order to since K is transposed
    kT = load_checked_2d(kT_ptrs, offs_k, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, KV_LEN)
    qk = tl.dot(q, kT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    pre_mod_scores = qk
    n = get_bounded_indices(offs_n2[None, :], KV_LEN if CHECK_BLOCK_BOUNDARY else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across N dim
    # that the M reads out of bounds prior to the last loop
    m = get_bounded_indices(offs_m2[:, None], Q_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qk",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        out="qk"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n2[None, :] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        {{ modification(
            subgraph_number=2,
            output_name="mask_mod_output",
            score="qk",
            b="off_z",
            h="off_hq",
            m="m",
            n="n",
        ) | indent_except_first(2) }}

        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, False)
        # apply mask for partial masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    p = tl.math.exp2(post_mod_scores - lse)
    # Compute dP and dS.
    # NB reversed order to since V is transposed
    vT = load_checked_2d(vT_ptrs, offs_v, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, V_HEAD_DIM, KV_LEN)

    dp = tl.dot(do, vT, input_precision=FLOAT32_PRECISION)
    ds = p * (dp - Di[:, None])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    {{ modification(
        subgraph_number=1,
        output_name = "grad_scores",
        score="pre_mod_scores",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        grad_score_mod="ds"
    ) | indent_except_first(1) }}
    if CHECK_BLOCK_BOUNDARY:
        grad_scores = tl.where(offs_n2[None, :] < KV_LEN, grad_scores, 0.0)

    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if WRITE_DQ:
        scatter_mask = offs_m2[:, None] < Q_LEN and offs_n2[None, :] < KV_LEN
        {{ modification(
            subgraph_number=3,
            output_name=None,
            mask="scatter_mask",
            score="pre_mod_scores",
            b="off_z",
            h="off_hq",
            m="m",
            n="n",
            grad_score_mod="ds"
        ) | indent_except_first(2) }}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = grad_scores

    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, False)
        # (grads) apply mask for partially unmasked block
        ds = tl.where(mask_mod_output, ds, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = ds.to(MATMUL_PRECISION)
    # Compute dQ.
    dq += tl.dot(ds, tl.trans(kT), input_precision=FLOAT32_PRECISION)

    return dq


@triton.jit
def bwd_dkdv_inner(
    {{gen_argdefs()}},
    Q, DO, DELTA, LSE, # pointers
    dk, dv, k, v,
    off_z, off_hq, offs_n1, offs_m1,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    {{gen_defines() | indent_except_first(1) }}
    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = {{size("Q", 2)}}
    KV_LEN = {{size("K", 2)}}

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    qT_ptrs = Q + offs_m1[None, :] * stride_qm + offs_k[:, None] * stride_qd
    do_ptrs = DO + offs_m1[:, None] * stride_dom + offs_v[None, :] * stride_dod
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    hi = tl.minimum(sparse_q_num_blocks * SPARSE_Q_MULTIPLE, tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1))

    if not IS_DIVISIBLE:
        if hi >= 1:
            for start_m in range(0, hi - 1):
                dk, dv = bwd_dkdv_block_mn(
                    {{gen_argdefs()}},
                    dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                    off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION, RCP_LN2,
                    IS_FULL_BLOCKS,
                )
                # Increment pointers.
                offset = get_offset_for_next_block(
                    start_m, q_indices, sparse_q_num_blocks,
                    SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1, BLOCKS_ARE_CONTIGUOUS
                )

                qT_ptrs += offset * stride_qm
                do_ptrs += offset * stride_dom

                offs_m1 += offset

            dk, dv = bwd_dkdv_block_mn(
                {{gen_argdefs()}},
                dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )
    else:
        for start_m in range(0, hi):
            dk, dv = bwd_dkdv_block_mn(
                {{gen_argdefs()}},
                dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
            # Increment pointers.
            offset = get_offset_for_next_block(
                start_m, q_indices, sparse_q_num_blocks,
                SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1, BLOCKS_ARE_CONTIGUOUS
            )

            qT_ptrs += offset * stride_qm
            do_ptrs += offset * stride_dom

            offs_m1 += offset

    return dk, dv


@triton.jit
def bwd_dkdv_block_mn(
    {{gen_argdefs()}},
    dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
    off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1) }}

    # NB reversed order since Q is transposed
    qT = load_checked_2d(qT_ptrs, offs_k, offs_m1, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, Q_LEN)
    # Load LSE before computing qk to reduce pipeline stall.
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN)
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(k, qT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    m = get_bounded_indices(offs_m1[None, :], Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across M dim
    # that the n reads out of bounds prior to the last loop
    n = get_bounded_indices(offs_n1[:, None], KV_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

    pre_mod_scores = qkT
    {{ modification(
        subgraph_number=0,
        output_name="post_mod_scores",
        score="qkT",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        out="qkT"
    ) | indent_except_first(1) }}

    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n1[:, None] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        {{ modification(
            subgraph_number=2,
            output_name="mask_mod_output",
            score="qkT",
            b="off_z",
            h="off_hq",
            m="m",
            n="n",
        ) | indent_except_first(2) }}
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, False)
        # (grads) apply mask for fully masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    pT = tl.math.exp2(post_mod_scores - lse[None, :])
    do = load_checked_2d(do_ptrs, offs_m1, offs_v, None, None, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM)
    # Compute dV.
    ppT = pT
    dv += tl.dot(ppT.to(MATMUL_PRECISION), do, input_precision=FLOAT32_PRECISION)
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN)
    # Compute dP and dS.
    dpT = tl.dot(v, tl.trans(do), input_precision=FLOAT32_PRECISION)
    dsT = pT * (dpT - Di[None, :])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    {{ modification(
        subgraph_number=1,
        output_name = "grad_scores",
        score="pre_mod_scores",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}

    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if not WRITE_DQ:
        idx_b = off_z
        idx_h = off_hq
        idx_m = m
        idx_n = n
        scatter_mask = offs_m1[None, :] < Q_LEN and offs_n1[:, None] < KV_LEN
        {{ modification(
            subgraph_number=3,
            output_name=None,
            mask="scatter_mask",
            score="pre_mod_scores",
            b="idx_b",
            h="idx_h",
            m="idx_m",
            n="idx_n",
            grad_score_mod="dsT"
        ) | indent_except_first(2) }}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if CHECK_BLOCK_BOUNDARY:
        grad_scores = tl.where(offs_n1[:, None] < KV_LEN, grad_scores, 0.0)

    dsT = grad_scores
    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, False)
        # (grads) apply mask for partially unmasked block
        dsT = tl.where(mask_mod_output, dsT, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dk += tl.dot(dsT.to(MATMUL_PRECISION), tl.trans(qT), input_precision=FLOAT32_PRECISION)

    return dk, dv
 """
    + compute_next_offset_func
    + get_bounded_indices_func
    + load_checked_2d,
)


def validate_joint_graph(joint_graph: torch.fx.Graph):
    """We do some pre lowering graph checks in order to raise nicer error messages"""
    for node in joint_graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.flex_lib.zeros_and_scatter.default
        ):
            for user in node.users:
                if user.op != "output":
                    raise NotImplementedError(
                        "Using multiple indexing operations on the same tensor that requires gradients "
                        "in a score_mod function is not currently supported. "
                        "This typically happens when indexing the same tensor multiple times, like:\n\n"
                        "    def score_mod(score, b, h, q_idx, kv_idx):\n"
                        "        return score + bias[q_idx] + bias[kv_idx]  # bias used twice!\n\n"
                        "A valid workaround is to clone() the tensors that will be indexed multiple times. For example:\n\n"
                        "    bias1 = bias.clone()\n"
                        "    def score_mod(score, b, h, q_idx, kv_idx):\n"
                        "        return score + bias[q_idx] + bias1[kv_idx]\n\n"
                        "Note that this solution will use additional memory."
                    )
    return


@dataclass(frozen=True)
class JointOutputResult:
    """Results from processing joint outputs."""

    grad_input: ComputedBuffer
    captured_grads_compute: list[ComputedBuffer]
    captured_grads: list[Optional[TensorBox]]
    mutated_grads: list[TensorBox]


def process_joint_outputs(
    all_joint_outputs: SubgraphResults, num_placeholders: int
) -> JointOutputResult:
    """Process joint outputs and extract various buffers needed for lowering

    Args:
        all_joint_outputs: List of all the outputs from build_subgraphs
        num_placeholders: The number of placeholder inputs, used to skip over unused backward compute buffers

    Returns:
        JointOutputResult containing processed buffers and gradients
    """
    assert isinstance(all_joint_outputs, list)
    assert all_joint_outputs[0] is not None, (
        "joint_subgraph_buffer is None - this is a bug!"
    )

    joint_buffer = all_joint_outputs[0]
    other_grads = all_joint_outputs[num_placeholders - 1 :]

    # outer_grads has the structure: Len(other_buffer_grads) if buffer doesn't require grad than it will be None
    # We only grab the buffers that require grad for inlining into kernel
    grads_compute = [buf for buf in other_grads if buf is not None]

    def get_out(buf):
        if buf is None:
            return None
        assert isinstance(buf, ComputedBuffer)
        assert buf.name is not None
        return TensorBox.create(V.graph.get_buffer(buf.name))

    grads_out = [get_out(x) for x in other_grads]
    mutated_grads = [buf for buf in grads_out if buf is not None]

    return JointOutputResult(
        grad_input=joint_buffer,
        captured_grads_compute=grads_compute,
        captured_grads=grads_out,
        mutated_grads=mutated_grads,
    )


# TODO: We probably also need a layout constraint?
@register_lowering(
    torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None
)
def flex_attention_backward(*args, **kwargs):
    (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ) = args
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    (
        query,
        key,
        value,
        grad_out,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            grad_out,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
    )

    device = query.get_device()
    dtype = query.get_dtype()
    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

    assert V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)), (
        f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
    )

    kernel_options = dict(kernel_options)
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.evaluate_static_shape(v)
        if isinstance(v, sympy.Symbol)
        else v
        for k, v in kernel_options.items()
    }
    kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
    if seq_len_q % 128 != 0 or seq_len_kv % 128 != 0:
        kernel_options.setdefault("IS_DIVISIBLE", False)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", True)

    fwd_placeholder_inps = [
        create_placeholder(name, dtype, device)
        for name, dtype in [
            ("score", dtype),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    fw_subgraph_buffer = build_subgraph_buffer(
        fwd_placeholder_inps + list(score_mod_other_buffers), fw_graph
    )

    joint_placeholder_inps = fwd_placeholder_inps + [
        create_placeholder("grad_score_mod", dtype, device)
    ]
    # Sometimes we have weird unused nodes here
    joint_graph.graph_module.graph.eliminate_dead_code()

    # It is hard to raise nice errors for some joint graphs during subgraph lowering
    # This lets us do some checks before attempting to lower
    validate_joint_graph(joint_graph.graph_module.graph)

    all_joint_outputs = build_subgraph_buffer(
        joint_placeholder_inps + list(score_mod_other_buffers),
        joint_graph,
    )

    joint_outputs = process_joint_outputs(
        all_joint_outputs, len(joint_placeholder_inps)
    )

    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )

    mask_graph_buffer = mask_graph_buffer

    # Construct layout with stride order matching K
    key_size = [Bq, Hkv, seq_len_kv, qk_head_dim]
    key_strides = infer_dense_strides(key_size, key.get_stride())

    layout_broadcasted_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        key_size,
        stride=[sympy.sympify(s) for s in key_strides],
    )

    # Create delta which will is needed for the bwd's kernel
    grad_lse_exp2 = lowerings[aten.mul](grad_logsumexp, 1 / math.log(2))
    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)
    delta = lowerings[aten.sub](delta, grad_lse_exp2)
    delta = ExternKernel.require_contiguous(delta)

    grad_lse_exp2, delta = maybe_realize([grad_lse_exp2, delta])

    # # see NOTE:[TritonTemplates with multiple outputs]
    grad_query = empty_like(query)

    # Construct output layout with stride order matching value
    value_size = [Bq, Hkv, seq_len_kv, v_head_dim]
    value_strides = infer_dense_strides(value_size, value.get_stride())

    broadcasted_grad_value = empty_strided(
        value_size,
        stride=[sympy.sympify(s) for s in value_strides],
        dtype=value.get_dtype(),
        device=value.get_device(),
    )

    kernel_options.setdefault("SM_SCALE", scale)

    # Determine GQA factor
    gqa_shared_heads = Hq // Hkv
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Inside of Triton kernel, only apply partial masking if partial blocks are computed.
    # full_kv_num_blocks is torch.zeros([1, 1, 1]) if partial blocks are not computed.
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices, full_q_num_blocks, full_q_indices = (
            empty(0, device=query.get_device()) for _ in range(4)
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_Q_BLOCK_SIZE)
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.evaluate_static_shape(SPARSE_KV_BLOCK_SIZE)

    choices: list[Any] = []
    configs: list[tuple[int, int, int, int]] = []
    configs.append(_get_default_config_bwd(query))
    if config.max_autotune:
        num_stages_list = [1, 3, 4, 5] if torch.version.hip is None else [1]
        configs.extend(
            [
                (BLOCK1, BLOCK2, w, s)
                for BLOCK1 in [32, 64]
                for BLOCK2 in [32, 64, 128]
                for w in ([4, 8] if BLOCK1 >= 128 or BLOCK2 >= 128 else [4])
                for s in num_stages_list
                if BLOCK2 % BLOCK1 == 0
            ]
        )
    original_kernel_options = kernel_options.copy()
    for BLOCK1, BLOCK2, num_warps, num_stages in configs:
        if (
            SPARSE_KV_BLOCK_SIZE % BLOCK1 != 0
            or SPARSE_Q_BLOCK_SIZE % BLOCK1 != 0
            or SPARSE_KV_BLOCK_SIZE % BLOCK2 != 0
            or SPARSE_Q_BLOCK_SIZE % BLOCK2 != 0
        ):
            continue

        # Performance tuning
        # Triton heuristics
        cur_kernel_options = original_kernel_options.copy()
        # Remove prefix for backward kernels options and delete forward kernel options.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("bwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("fwd_"):
                cur_kernel_options.pop(k)
        cur_kernel_options.setdefault("num_warps", num_warps)
        cur_kernel_options.setdefault("num_stages", num_stages)

        cur_kernel_options.setdefault("BLOCK_M1", BLOCK1)
        cur_kernel_options.setdefault("BLOCK_N1", BLOCK2)
        cur_kernel_options.setdefault("BLOCK_M2", BLOCK2)
        cur_kernel_options.setdefault("BLOCK_N2", BLOCK1)
        # Blocksparse options
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

        flex_attention_backward_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                delta,
                grad_out,
                grad_query,
                broadcasted_grad_value,
                kv_num_blocks,
                kv_indices,
                q_num_blocks,
                q_indices,
                full_kv_num_blocks,
                full_kv_indices,
                full_q_num_blocks,
                full_q_indices,
            ],
            layout=layout_broadcasted_k,  # We use store_output only for grad_key
            subgraphs=[
                fw_subgraph_buffer,
                joint_outputs.grad_input,
                mask_graph_buffer,
                joint_outputs.captured_grads_compute,
            ],
            mutated_inputs=[
                grad_query,
                broadcasted_grad_value,
                *joint_outputs.mutated_grads,
            ],
            call_sizes=query.get_size() + key.get_size()[1:3],
            **cur_kernel_options,
        )
    inputs_for_autotuning = (
        [
            query,
            key,
            value,
            logsumexp,
            delta,
            grad_out,
            grad_query,
            broadcasted_grad_value,
            kv_num_blocks,
            kv_indices,
            q_num_blocks,
            q_indices,
            full_kv_num_blocks,
            full_kv_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
        + joint_outputs.mutated_grads
    )
    input_gen_fns = {
        8: create_num_blocks_fake_generator(kv_indices),  # kv_num_blocks
        9: create_indices_fake,
        10: create_num_blocks_fake_generator(q_indices),  # q_num_blocks
        11: create_indices_fake,
        12: create_num_blocks_fake_generator(full_kv_indices),  # full_kv_num_blocks
        13: create_indices_fake,
        14: create_num_blocks_fake_generator(full_q_indices),  # full_q_num_blocks
        15: create_indices_fake,
    }

    broadcasted_grad_key = autotune_select_algorithm(
        "flex_attention_backward",
        choices,
        inputs_for_autotuning,
        layout_broadcasted_k,
        input_gen_fns=input_gen_fns,
    )  # [Bq, Hkv, seq_len_kv, k_head_dim]

    if V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv)):
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value
    else:
        assert V.graph.sizevars.evaluate_expr(sympy.Gt(Bq, 1) & sympy.Eq(Bkv, 1)), (
            f"Bq and Bkv must broadcastable. "
            f"Got Bq={V.graph.sizevars.evaluate_expr(Bq)} "
            f"and Bkv={V.graph.sizevars.evaluate_expr(Bkv)}"
        )
        grad_key = lowerings[aten.sum](broadcasted_grad_key, axis=0, keepdims=True)
        grad_value = lowerings[aten.sum](broadcasted_grad_value, axis=0, keepdims=True)

    return (grad_query, grad_key, grad_value, tuple(joint_outputs.captured_grads))
