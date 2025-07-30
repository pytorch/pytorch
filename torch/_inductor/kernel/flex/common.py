# mypy: allow-untyped-defs
"""Common utilities and functions for flex attention kernels"""

import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map

from ...ir import (
    ComputedBuffer,
    ExternKernel,
    FixedLayout,
    FlexibleLayout,
    get_fill_order,
    InputBuffer,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    Scatter,
    ShapeAsConstantBuffer,
    StorageBox,
    Subgraph,
    TensorBox,
)
from ...lowering import (
    _full,
    check_and_broadcast_indices,
    expand,
    index_output_size_and_inner_fn,
    to_dtype,
)
from ...select_algorithm import realize_inputs


SubgraphResults = Union[list[Optional[ComputedBuffer]], Optional[ComputedBuffer]]


def zeros_and_scatter_lowering(shape: list[int], indices, values):
    """To support backwards on captured buffers we register a specific lowering for our specific custom up"""
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


def get_fwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults, mask_graph_buffer: SubgraphResults
) -> list[Optional[ComputedBuffer]]:
    subgraph_buffer = (
        subgraph_buffer if isinstance(subgraph_buffer, Sequence) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        mask_graph_buffer
        if isinstance(mask_graph_buffer, Sequence)
        else [mask_graph_buffer]
    )
    return [*subgraph_buffer, *mask_graph_buffer]


def build_subgraph_module_buffer(
    args: list[Union[TensorBox, ShapeAsConstantBuffer]],
    graph_module: torch.fx.GraphModule,
) -> SubgraphResults:
    """This function's goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that are passed into the subgraph. Contains both fixed and lifted inputs.
        subgraph: The Subgraph ir for which to produce the output node
    """
    # This one we gotta keep lazy
    from ...subgraph_lowering import PointwiseSubgraphLowering

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
        device = output_buffer.data.get_device()
        assert device is not None
        subgraph_buffer = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=output_buffer.data.get_dtype(),
                size=output_buffer.data.get_size(),
            ),
            data=output_buffer.data.data,  # type: ignore[arg-type]
        )
        return subgraph_buffer

    return tree_map(convert_output_node_to_buffer, pw_subgraph.graph_outputs)


def build_subgraph_buffer(
    args: list[Union[TensorBox, ShapeAsConstantBuffer]], subgraph: Subgraph
) -> SubgraphResults:
    return build_subgraph_module_buffer(args, subgraph.graph_module)


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


def create_placeholder(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    size: Optional[list[int]] = None,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
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


def create_indices_fake(x) -> torch.Tensor:
    """Create a fake indices that is used for autotuning."""
    size = [V.graph.sizevars.size_hint(i) for i in x.get_size()]
    indices = torch.arange(0, size[-1], dtype=x.get_dtype(), device=x.get_device())
    indices = indices.expand(size).contiguous()
    return indices


def create_num_blocks_fake_generator(sparse_indices):
    """Create a fake num_blocks that is used for autotuning.

    The idea here is that we need to create a real tensor with real data
    that's representative for benchmarking.
    For example, returning all zeros for the `kv_num_blocks` input would mean
    that we are computing 0 blocks for each row, which would provide bogus
    autotuning results.

    In this case, we choose to use min(16, max_block) blocks, because I
    (Horace) think it'll probably result in pretty representative performance.
    If it's too short then prefetching won't help. If it's too long then
    autotuning will take longer for no good reason.
    """

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


def contiguous_last_dim(x):
    """Ensure that realized IR node has a contiguous stride in the last dimension."""
    strides = x.maybe_get_stride()
    if strides and strides[-1] != 1:
        contiguous_stride_order = list(reversed(range(len(x.get_size()))))
        return ExternKernel.require_stride_order(x, contiguous_stride_order)
    return x


def set_head_dim_values(
    kernel_options: dict[str, Any], qk_head_dim, v_head_dim, graph_sizevars
):
    """
    Mutates kernel options, adding head dimension calculations.

    Args:
        kernel_options: Dictionary to populate with options
        qk_head_dim: Query/Key head dimension
        v_head_dim: Value head dimension
        graph_sizevars: Graph size variables object with guard_int method

    """
    # QK dimensions
    qk_head_dim_static = graph_sizevars.guard_int(qk_head_dim)
    kernel_options.setdefault("QK_HEAD_DIM", qk_head_dim_static)
    kernel_options.setdefault(
        "QK_HEAD_DIM_ROUNDED", next_power_of_two(qk_head_dim_static)
    )

    # V dimensions
    v_head_dim_static = graph_sizevars.guard_int(v_head_dim)
    kernel_options.setdefault("V_HEAD_DIM", v_head_dim_static)
    kernel_options.setdefault(
        "V_HEAD_DIM_ROUNDED", next_power_of_two(v_head_dim_static)
    )

    # Safety flag
    kernel_options.setdefault(
        "SAFE_HEAD_DIM",
        is_power_of_2(qk_head_dim_static) and is_power_of_2(v_head_dim_static),
    )


def is_power_of_2(n):
    return n != 0 and ((n & (n - 1)) == 0)


def next_power_of_two(n):
    if n <= 0:
        return 1
    return 2 ** math.ceil(math.log2(n))


# ---- Common Template Strings ----
compute_forward_block_mn = r"""
@triton.jit
def forward_block_mn(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr, desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    kv_offset,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    {{gen_defines() | indent_except_first(1)}}

    # -- load k --
    # NB reversed order to since K is transposed
    {%- if USE_TMA %}
    k = tl.load_tensor_descriptor(
        desc_k,
        [kv_start + kv_offset, 0],
    )
    {%- else %}
    k = load_checked_block(K_block_ptr, SAFE_HEAD_DIM, IS_DIVISIBLE)
    {%- endif %}

    if USE_TMA:
        k = tl.trans(k)
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
    {%- if USE_TMA %}
    v = tl.load_tensor_descriptor(
        desc_v,
        [kv_start + kv_offset, 0],
    )
    {%- else %}
    v = load_checked_block(V_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    {%- endif %}
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

"""

compute_forward_inner = r"""
@triton.jit
def forward_inner(
    {{gen_argdefs()}},
    q, K_block_ptr, V_block_ptr,
    desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
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

    kv_offset = 0

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        # Here IS_DIVISIBLE acts are the start_n = tl.multiple_of(start_n, BLOCK_N) from triton_fused_attention.
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                {{gen_argdefs()}},
                q, K_block_ptr, V_block_ptr, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
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
                q, K_block_ptr, V_block_ptr, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )



        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        offs_n = offs_n + offset
        kv_offset += offset
        if not USE_TMA:
            K_block_ptr = tl.advance(K_block_ptr, (0, offset))
            V_block_ptr = tl.advance(V_block_ptr, (offset, 0))


    return acc, l_i, m_i

"""

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
