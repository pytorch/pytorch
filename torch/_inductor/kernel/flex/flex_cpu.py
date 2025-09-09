# mypy: allow-untyped-defs
"""CPU-specific implementations for flex attention"""

import copy
import os
import sys
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges

from ...codegen.cpp_flex_attention_template import CppFlexAttentionTemplate
from ...ir import Buffer, FixedLayout, TensorBox
from ...select_algorithm import autotune_select_algorithm
from .common import (
    build_subgraph_buffer,
    build_subgraph_module_buffer,
    contiguous_last_dim,
    create_placeholder,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    maybe_realize,
)


def check_cpu_supported():
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
    """CPP based template for flex attention for x86 CPUs"""
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

    # We don't know the concrete value of cur_qSplitSize and cur_kvSplitSize during the compilation.
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

    # CPU kernel requires last dim to be contiguous
    query, key, value = map(contiguous_last_dim, [query, key, value])

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
        input_nodes += [full_kv_indices]
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
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
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
        q_block_size=SPARSE_Q_BLOCK_SIZE,
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

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    res.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    res.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )

    return (res,)
