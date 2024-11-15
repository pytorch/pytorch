# mypy: allow-untyped-defs
""" Triton Implementation of the flex_attention Kernel"""

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._pytree import tree_map

from .. import config
from ..ir import (
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
    empty_strided,
    expand,
    index_output_size_and_inner_fn,
    lowerings,
    register_lowering,
    to_dtype,
)
from ..select_algorithm import autotune_select_algorithm, realize_inputs, TritonTemplate


log = logging.getLogger(__name__)
aten = torch.ops.aten
Expr = sympy.Expr


def zeros_and_scatter_lowering(shape: List[int], indices, values):
    grad = _full(0, values.get_device(), values.get_dtype(), shape)
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
    scatter = Scatter(
        device=grad.get_device(),
        dtype=grad.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=inner_fn,
        scatter_mode="atomic_add",
    )

    buffer = ComputedBuffer(
        name=grad.data.data.name,
        layout=MutationLayoutSHOULDREMOVE(grad),
        data=scatter,
    )
    return (buffer, grad)


def construct_strides(
    sizes: Sequence[int],
    fill_order: Sequence[int],
) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    assert len(sizes) == len(
        fill_order
    ), "Length of sizes must match the length of the fill order"
    strides = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, ceil_div(n_queries, query_block_size), 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    import triton

    return (triton.cdiv(num_queries, meta["BLOCK_M"]), batch_size * q_heads, 1)


def create_placeholder(
    name: str, dtype: torch.dtype, device: torch.device
) -> TensorBox:
    """Creates a placeholder input buffers for producing subgraph_output."""
    input_buffer = InputBuffer(name=name, layout=FixedLayout(device, dtype, [], []))
    return TensorBox.create(input_buffer)


def maybe_realize(args: List[Optional[IRNode]]):
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
    if torch.get_float32_matmul_precision() == "highest" or torch.version.hip:
        return "'ieee'"
    else:
        return "'tf32'"


@dataclass(frozen=True)
class SubgraphOutput:
    """When building the subgraph buffers we fall into 1 of two cases:
    1. The subgraph returns a single output node, this happens in the forward and
        backward pass when there are no input_buffers that require grad
    2. The subgraph has multiple outputs (i.e. input_buffers that require grad)
        In this case we want to return these output_nodes from the lowering
    """

    compute_buffer: ComputedBuffer
    output_node: Optional[IRNode] = None

    def __post_init__(self):
        assert isinstance(self.compute_buffer, ComputedBuffer), (
            "The compute buffer for the subgraph output must be a TensorBox, but got: ",
            type(self.compute_buffer),
        )
        if self.output_node is not None:
            assert isinstance(self.output_node, IRNode), (
                "The output node for the subgraph output must be an IRNode, but got: ",
                type(self.output_node),
            )


def build_subgraph_buffer(
    args: List[TensorBox],
    subgraph: Subgraph,
) -> Union[List[SubgraphOutput], SubgraphOutput]:
    """This function's goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph. Contains both fixed and lifted inputs.
        subgraph: The Subgraph ir for which to produce the output node
    """
    cnt = 0
    env = {}
    for node in subgraph.graph_module.graph.nodes:
        # There are two classes of placeholder inpts that we need
        # to handle differently. For the first n_scalar_inps inputs
        # we expect that these placeholders were generated by the make_fx call
        # in the flex Attention HOP. So we need to create a new placeholder
        # TensorBox for each of these inputs. For the rest of the inputs we
        # expect that these are lifted inputs that fill up the '*other_buffers'
        # tuple and already have corresponding TensorBoxes passed in as args.
        with V.graph.set_current_node(node):
            if node.op == "placeholder":
                env[node] = args[cnt]
                cnt += 1
            elif node.op == "call_function":
                # For call_function we use the default lowerings and pass in the
                # already created TensorBoxes as args
                args, kwargs = tree_map(
                    lambda x: env[x] if x in env else x, (node.args, node.kwargs)
                )
                flex_lowering = (
                    lowerings[node.target]
                    if node.target._opname != "zeros_and_scatter"
                    else zeros_and_scatter_lowering
                )
                env[node] = flex_lowering(*args, **kwargs)
            elif node.op == "output":

                def convert_output_node_to_buffer(output):
                    if output is None:
                        return None
                    output_node = output
                    output_buffer = env[output_node]
                    if isinstance(output_buffer, tuple):
                        # These nodes are coming from the output of zeros_and_scatter
                        return SubgraphOutput(
                            compute_buffer=output_buffer[0],
                            output_node=output_buffer[1],
                        )
                    assert isinstance(output_buffer, TensorBox), (
                        "The output node  for flex attention's subgraph must be a TensorBox, but got: ",
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
                    return SubgraphOutput(compute_buffer=subgraph_buffer)

                # node.args[0] is either a single element or a list of elements
                # representing all outputs of the function.
                return tree_map(convert_output_node_to_buffer, node.args[0])

    raise ValueError("FlexAttention was passed a subgraph with no output node!")


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
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

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
        block_shape=(BLOCK_M, QK_HEAD_DIM),
        order=(1, 0)
    )

    # load q: it stays in SRAM throughout the inner loop.
    if IS_DIVISIBLE:
        q = tl.load(Q_block_ptr)
    else:
        # boundary check is not free, so we only do it when necessary.
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option = "zero")

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
        block_shape=(QK_HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(KV_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM),
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
            block_shape=(QK_HEAD_DIM, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM),
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
    idx_d = tl.arange(0, V_HEAD_DIM)[None, :]

    mask = idx_m < Q_LEN
    # TODO generalize and add proper mask support
    {{store_output(("idx_zq", "idx_hq", "idx_m", "idx_d"), "acc", "mask")}}

    # TODO dont want to write this if we dont require grad
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
    if IS_DIVISIBLE:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option = "zero")
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    if CHECK_BLOCK_BOUNDARY:
        # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
        # which is larger than the actual number of elements. To avoid access memory out of bound,
        # we need to mask out the elements that are out of Q_LEN & KV_LEN.
        m = offs_m % Q_LEN
        n = offs_n % KV_LEN
    else:
        m = offs_m
        n = offs_n

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
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, float("-inf"))
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    # TODO: In the case that score_mod is linear, this can be LICMed
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

    if IS_DIVISIBLE:
        v = tl.load(V_block_ptr)
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
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
    + compute_forward_block_mn,
)


def _use_flex_decoding(query, kernel_options):
    # Decide which kernel to use, return true if use flex decoding kernel.
    return (
        not kernel_options.get("FORCE_USE_FLEX_ATTENTION", False)
    ) and V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-2], 128))


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


def _get_default_config_fwd(query) -> Tuple[int, int, int, int]:
    dtype = query.get_dtype()
    head_dim = query.get_size()[-1]
    default_config = None

    if head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _h100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (8, 0):  # A100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _a100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.version.hip:
        if dtype == torch.float32:
            default_config = (64, 64, 4, 1)
        else:
            default_config = (128, 64, 8, 1)
        default_config = _rocm_default_config.get((dtype, head_dim), default_config)
    else:  # modest hardware or extremely large head_dim
        if dtype == torch.float32:
            default_config = (32, 16, 4, 3)
        else:
            default_config = (64, 32, 4, 3)

    return default_config


def _get_default_config_bwd(query) -> Tuple[int, int, int, int]:
    head_dim = query.get_size()[-1]
    dtype = query.get_dtype()

    if dtype == torch.float32:
        return (16, 16, 4, 1)
    if head_dim <= 256 and torch.version.hip:
        if head_dim == 64:
            return (64, 64, 4, 1)
        elif head_dim == 128:
            return (64, 128, 4, 1)
        else:
            return (64, 64, 4, 1)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if head_dim == 64:
            return (64, 64, 4, 3)
        elif head_dim == 128:
            return (64, 128, 8, 3)
        else:
            return (64, 64, 4, 2)
    elif torch.cuda.get_device_capability() >= (8, 0):  # A100
        if head_dim == 64:
            return (32, 128, 4, 3)
        elif head_dim == 128:
            return (64, 128, 8, 3)
        else:
            return (64, 64, 4, 2)
    else:  # modest hardware or extremely large head_dim
        return (16, 16, 4, 1)


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
        num_blocks_for_autotuning = min(16, sparse_indices.shape[-1])
        return torch.full(
            x.get_size(),
            int(num_blocks_for_autotuning),
            dtype=x.get_dtype(),
            device=x.get_device(),
        )

    return create_num_blocks_fake


def create_indices_fake(x) -> torch.Tensor:
    indices = torch.arange(
        0, int(x.get_size()[-1]), dtype=x.get_dtype(), device=x.get_device()
    )
    indices = indices.expand(x.get_size()).contiguous()
    return indices


from torch._inductor.kernel.flex_decoding import create_flex_decoding_kernel


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
    (
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
    assert isinstance(subgraph_buffer, SubgraphOutput)
    subgraph_buffer = subgraph_buffer.compute_buffer
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
    assert isinstance(mask_graph_buffer, SubgraphOutput)
    mask_graph_buffer = mask_graph_buffer.compute_buffer
    kernel_options = dict(kernel_options)
    kernel_options.setdefault("FLOAT32_PRECISION", get_float32_precision())
    if _use_flex_decoding(query, kernel_options):
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
    assert V.graph.sizevars.evaluate_expr(
        sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)
    ), f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
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
    fill_order = get_fill_order(query.get_stride())
    out_strides = construct_strides(out_size, fill_order)

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, Hq, seq_len_q, v_head_dim],
        stride=out_strides,
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
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim)

    choices: List[Any] = []
    configs: List[Tuple[int, int, int, int]] = []
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
    assert V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_q, sympy.Mul(kv_indices.get_size()[-2], SPARSE_Q_BLOCK_SIZE))
    ), "Q seqlen must be smaller than the block_mask size in the Q dimension, considering pass a larger block_mask."
    assert V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_kv, sympy.Mul(kv_indices.get_size()[-1], SPARSE_KV_BLOCK_SIZE))
    ), "KV seqlen must be smaller than the block_mask size in the KV dimension, considering pass a larger block_mask."

    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    original_kernel_options = kernel_options.copy()
    for BLOCK_M, BLOCK_N, num_warps, num_stages in configs:
        if SPARSE_KV_BLOCK_SIZE % BLOCK_N != 0 or SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0:
            if len(configs) == 1:
                raise ValueError(
                    f"Q and KV block size must be divisible by BLOCK_M and BLOCK_N. We"
                    f"got Q_BLOCK_SIZE={SPARSE_Q_BLOCK_SIZE} and KV_BLOCK_SIZE={SPARSE_KV_BLOCK_SIZE}."
                )
            continue
        # Work around https://github.com/pytorch/pytorch/issues/129625
        if num_stages == 2:
            continue

        cur_kernel_options = original_kernel_options.copy()
        # Performance tuning
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
            num_stages=num_stages,
            num_warps=num_warps,
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
    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

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

        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM], dtype=tl.float32)

        start_m2 = start_m2_block * BLOCK_M2
        offs_m2 = start_m2 + tl.arange(0, BLOCK_M2)

        # load Q and do: they stay in SRAM throughout the inner loop.
        if IS_DIVISIBLE:
            q = tl.load(Q2 + offs_m2[:, None] * stride_qm + offs_k[None, :] * stride_qd)
            do = tl.load(DO2 + offs_m2[:, None] * stride_dom + offs_v[None, :] * stride_dod)
        else:
            q = tl.load(Q2 + offs_m2[:, None] * stride_qm + offs_k[None, :] * stride_qd, mask=offs_m2[:, None] < Q_LEN)
            do = tl.load(DO2 + offs_m2[:, None] * stride_dom + offs_v[None, :] * stride_dod, mask=offs_m2[:, None] < Q_LEN)

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
        if IS_DIVISIBLE:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=offs_m2[:, None] < Q_LEN)
    else:
        # THIS BLOCK DOES DK & DV
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

        pid_mask = pid // SPARSE_KV_MULTIPLE

        stride_q_num_blks_h = {{stride("Q_NUM_BLKS", 1)}}
        stride_q_idx_h = {{stride("Q_IDX", 1)}}
        stride_q_idx_n = {{stride("Q_IDX", 2)}}

        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, QK_HEAD_DIM], dtype=tl.float32)

        start_n1 = pid * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)

        # load K and V: they stay in SRAM throughout the inner loop.
        if IS_DIVISIBLE:
            k = tl.load(K + offs_n1[:, None] * stride_kn + offs_k[None, :] * stride_kd)
            v = tl.load(V + offs_n1[:, None] * stride_vn + offs_v[None, :] * stride_vd)
        else:
            k = tl.load(K + offs_n1[:, None] * stride_kn + offs_k[None, :] * stride_kd, mask=offs_n1[:, None] < KV_LEN)
            v = tl.load(V + offs_n1[:, None] * stride_vn + offs_v[None, :] * stride_vd, mask=offs_n1[:, None] < KV_LEN)
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

        if IS_DIVISIBLE:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=index_n < KV_LEN)

        dk *= SM_SCALE
        mask = index_n < KV_LEN

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

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

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
                    off_z, off_hq, offs_m2, offs_n2,
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
                off_z, off_hq, offs_m2, offs_n2,
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
                off_z, off_hq, offs_m2, offs_n2,
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
    off_z, off_hq, offs_m2, offs_n2,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1)}}

    if IS_DIVISIBLE:
        kT = tl.load(kT_ptrs)
    else:
        kT = tl.load(kT_ptrs, mask=offs_n2[None, :] < KV_LEN)
    qk = tl.dot(q, kT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    pre_mod_scores = qk
    if CHECK_BLOCK_BOUNDARY:
        m = offs_m2[:, None] % Q_LEN
        n = offs_n2[None, :] % KV_LEN
    else:
        m = offs_m2[:, None]
        n = offs_n2[None, :]
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
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, float("-inf"))
        # apply mask for partial masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    p = tl.math.exp2(post_mod_scores - lse)
    # Compute dP and dS.
    if IS_DIVISIBLE:
        vT = tl.load(vT_ptrs)
    else:
        vT = tl.load(vT_ptrs, mask=offs_n2[None, :] < KV_LEN)
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

    ds = grad_scores

    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, float("-inf"))
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

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

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
                    off_z, off_hq, offs_n1, offs_m1,
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
                off_z, off_hq, offs_n1, offs_m1,
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
                off_z, off_hq, offs_n1, offs_m1,
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
    off_z, off_hq, offs_n1, offs_m1,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    {{gen_defines() | indent_except_first(1) }}

    # Load LSE before computing qk to reduce pipeline stall.
    if IS_DIVISIBLE:
        qT = tl.load(qT_ptrs)
        lse = tl.load(LSE + offs_m1)
    else:
        qT = tl.load(qT_ptrs, mask=offs_m1[None, :] < Q_LEN)
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN)
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(k, qT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    if CHECK_BLOCK_BOUNDARY:
        m = offs_m1[None, :] % Q_LEN
        n = offs_n1[:, None] % KV_LEN
    else:
        m = offs_m1[None, :]
        n = offs_n1[:, None]
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
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, float("-inf"))
        # (grads) apply mask for fully masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    pT = tl.math.exp2(post_mod_scores - lse[None, :])
    if IS_DIVISIBLE:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=offs_m1[:, None] < Q_LEN)
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
    {{ modification(
        subgraph_number=3,
        output_name=None,
        score="pre_mod_scores",
        b="off_z",
        h="off_hq",
        m="m",
        n="n",
        grad_score_mod="dsT"
    ) | indent_except_first(1) }}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if CHECK_BLOCK_BOUNDARY:
        grad_scores = tl.where(offs_n1[:, None] < KV_LEN, grad_scores, 0.0)

    dsT = grad_scores
    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, float("-inf"))
        # (grads) apply mask for partially unmasked block
        dsT = tl.where(mask_mod_output, dsT, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dk += tl.dot(dsT.to(MATMUL_PRECISION), tl.trans(qT), input_precision=FLOAT32_PRECISION)

    return dk, dv
 """
    + compute_next_offset_func,
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

    assert V.graph.sizevars.evaluate_expr(
        sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)
    ), f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
    B = Bq

    kernel_options = dict(kernel_options)
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
    assert isinstance(fw_subgraph_buffer, SubgraphOutput)
    fw_subgraph_buffer = fw_subgraph_buffer.compute_buffer

    joint_placeholder_inps = fwd_placeholder_inps + [
        create_placeholder("grad_score_mod", dtype, device)
    ]
    all_joint_outputs = build_subgraph_buffer(
        joint_placeholder_inps + list(score_mod_other_buffers), joint_graph
    )
    assert isinstance(all_joint_outputs, List)
    joint_subgraph_buffer = all_joint_outputs[0]
    joint_subgraph_buffer = joint_subgraph_buffer.compute_buffer
    score_mod_other_buffer_grads = all_joint_outputs[len(joint_placeholder_inps) - 1 :]

    score_mod_other_buffer_grads_compute = [
        buf.compute_buffer for buf in score_mod_other_buffer_grads
    ]
    score_mod_other_buffer_grads_out = [
        buf.output_node for buf in score_mod_other_buffer_grads
    ]

    mutated_other_buffer_grads = [
        buffer for buffer in score_mod_other_buffer_grads_out if buffer is not None
    ]

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
    assert isinstance(mask_graph_buffer, SubgraphOutput)
    mask_graph_buffer = mask_graph_buffer.compute_buffer

    layout_broadcasted_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        [Bq, Hkv, seq_len_kv, qk_head_dim],
        key.get_stride(),
    )

    # Create delta which will is needed for the bwd's kernel
    grad_lse_exp2 = lowerings[aten.mul](grad_logsumexp, 1 / math.log(2))
    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)
    delta = lowerings[aten.sub](delta, grad_lse_exp2)
    delta = ExternKernel.require_contiguous(delta)

    grad_lse_exp2, delta = maybe_realize([grad_lse_exp2, delta])

    # see NOTE:[TritonTemplates with multiple outputs]
    grad_query = empty_strided(
        query.get_size(), query.get_stride(), dtype=dtype, device=device
    )
    broadcasted_grad_value = empty_strided(
        (Bq, *value.get_size()[1:]),
        value.get_stride(),
        dtype=dtype,
        device=device,
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
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim)

    choices: List[Any] = []
    configs: List[Tuple[int, int, int, int]] = []
    configs.append(_get_default_config_bwd(query))
    if config.max_autotune:
        num_stages_list = [1, 3, 4, 5] if torch.version.hip is None else [1]
        configs.extend(
            [
                (BLOCK1, BLOCK2, w, s)
                for BLOCK1 in [32, 64]
                for BLOCK2 in [32, 64, 128]
                for w in [4, 8]
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
        cur_kernel_options = original_kernel_options.copy()
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
                joint_subgraph_buffer,
                mask_graph_buffer,
                score_mod_other_buffer_grads_compute,
            ],
            mutated_inputs=[
                grad_query,
                broadcasted_grad_value,
                *mutated_other_buffer_grads,
            ],
            call_sizes=query.get_size() + key.get_size()[1:3],
            num_stages=num_stages,
            num_warps=num_warps,
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
        assert V.graph.sizevars.evaluate_expr(
            sympy.Gt(Bq, 1) & sympy.Eq(Bkv, 1)
        ), f"Bq and Bkv must broadcastable. Got Bq={V.graph.sizevars.evaluate_expr(Bq)} and Bkv={V.graph.sizevars.evaluate_expr(Bkv)}"  # noqa: B950
        grad_key = lowerings[aten.sum](broadcasted_grad_key, axis=0, keepdims=True)
        grad_value = lowerings[aten.sum](broadcasted_grad_value, axis=0, keepdims=True)

    # import fbvscode; fbvscode.set_trace()
    return (
        grad_query,
        grad_key,
        grad_value,
        tuple(score_mod_other_buffer_grads_out),
    )
