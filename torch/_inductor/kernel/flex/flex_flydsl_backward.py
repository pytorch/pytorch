# mypy: allow-untyped-defs

import math
from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE

from ...codegen.flydsl.flydsl_template import FlyDSLTemplate
from ...codegen.flydsl.flydsl_utils import runtime_available
from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import (
    constant_pad_nd,
    empty_strided,
    full,
    permute,
    slice_,
    to_dtype,
    view,
)
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import V
from ..vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
    CAPTURE_GRAD_FULL,
    CAPTURE_GRAD_INVARIANT,
    CAPTURE_GRAD_KEY,
    CAPTURE_GRAD_QUERY,
    CAPTURE_GRAD_RELATIVE,
    CAPTURE_GRAD_UNUSED,
    capture_grad_reduction_shape,
    capture_grad_workspace_shape,
    choose_dq_partitions,
    classify_mask_traversal,
    MASK_TRAVERSAL_BLOCK_LIST,
    MAX_DQ_WORKSPACE_BYTES,
)
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    freeze_irnodes,
    infer_dense_strides,
    load_flex_template,
    maybe_realize,
)
from .flex_flash_attention import is_trivial_mask_graph
from .flex_flydsl_attention import (
    _create_dense_metadata,
    _get_block_mask_metadata,
    _get_flydsl_common_config,
    _get_mask_program_config,
    _kernel_capture_buffers,
)
from .flex_flydsl_score import (
    classify_capture_gradient,
    FlyDSLCaptureGrad,
    FlyDSLScoreProgram,
)


flex_flydsl_backward_template = FlyDSLTemplate(
    name="flex_flydsl_backward", source=load_flex_template("flydsl_backward")
)

_METADATA_BLOCK_SIZE = 128
# Workspace slots for trainable score captures in the backward kernel.
_MAX_CAPTURE_GRADS = 2


def _get_flydsl_flex_attention_backward_config(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    query: TensorBox,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
    *,
    key: TensorBox,
    value: TensorBox,
    out: TensorBox | None = None,
    grad_out: TensorBox | None = None,
    grad_logsumexp: TensorBox | None = None,
    mask_mod_other_buffers: Sequence[TensorBox] | None = None,
    kv_num_blocks: TensorBox | None = None,
    kv_indices: TensorBox | None = None,
    full_kv_num_blocks: TensorBox | None = None,
    full_kv_indices: TensorBox | None = None,
    scale: float | None = None,
    sparse_q_block_size: int | None = None,
    sparse_kv_block_size: int | None = None,
    dq_accum_fp32: bool = True,
    block_mask_is_dense: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    score_mod_other_buffers = score_mod_other_buffers or ()
    mask_mod_other_buffers = mask_mod_other_buffers or ()

    mask_config, reason = _get_mask_program_config(mask_graph, mask_mod_other_buffers)
    if mask_config is None:
        return None, reason

    common_config, reason = _get_flydsl_common_config(
        query=query,
        key=key,
        value=value,
        subgraph=fw_subgraph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        extra_strides=(("OUT_STRIDE", out), ("DO_STRIDE", grad_out)),
        allow_mask_mod_buffers=bool(mask_config["MASK_BUFFER_COUNT"]),
        allow_strided_bhsd=True,
    )
    if common_config is None:
        return None, f"FlyDSL flex bwd {reason}"

    if scale is None:
        return None, "FlyDSL flex bwd requires static shapes and scale"

    try:
        b, h, sq, dqk = [V.graph.sizevars.guard_int(item) for item in query.get_size()]
        bk, hk, sk, dk = [V.graph.sizevars.guard_int(item) for item in key.get_size()]
        bv, hv, sv, dv = [V.graph.sizevars.guard_int(item) for item in value.get_size()]
        out_shape = (
            None
            if out is None
            else [V.graph.sizevars.guard_int(item) for item in out.get_size()]
        )
        grad_out_shape = (
            None
            if grad_out is None
            else [V.graph.sizevars.guard_int(item) for item in grad_out.get_size()]
        )
        block_m = V.graph.sizevars.guard_int(sparse_q_block_size)
        block_n = V.graph.sizevars.guard_int(sparse_kv_block_size)
        scale_value = float(scale)
    except (AttributeError, TypeError, ValueError):
        return None, "FlyDSL flex bwd requires static shapes and scale"

    if (b, h, sq, dqk) != (bk, hk, sk, dk):
        return None, "FlyDSL flex bwd currently supports MHA with matching Q/K shapes"
    if (bv, hv, sv) != (b, h, sq):
        return None, "FlyDSL flex bwd currently supports MHA with matching B/H/S"
    if (dqk, dv) not in ((128, 128), (192, 128)):
        return (
            None,
            "FlyDSL flex bwd supports only (QK head dim, V head dim) "
            "= (128, 128) or (192, 128)",
        )
    expected_out_shape = [b, h, sq, dv]
    if out_shape is not None and out_shape != expected_out_shape:
        return None, "FlyDSL flex bwd requires OUT shape [B, H, S, Dv]"
    if grad_out_shape is not None and grad_out_shape != expected_out_shape:
        return None, "FlyDSL flex bwd requires grad_out shape [B, H, S, Dv]"
    if out is not None and out.get_dtype() != torch.bfloat16:
        return None, "FlyDSL flex bwd requires BF16 OUT"
    if grad_out is not None and grad_out.get_dtype() != torch.bfloat16:
        return None, "FlyDSL flex bwd requires BF16 grad_out"
    if block_m != 128 or block_n != 128:
        return None, "FlyDSL flex bwd currently requires Q/KV block size 128"
    if sq % 128:
        return None, "FlyDSL flex bwd requires sequence length divisible by 128"
    if sq > 16384:
        return None, "FlyDSL flex bwd currently supports sequence length <= 16384"
    if grad_logsumexp is not None:
        return None, "FlyDSL flex bwd does not support gradients through LSE aux"
    workspace_element_bytes = 4 if dq_accum_fp32 else 2
    if b * h * sq * dqk * workspace_element_bytes > MAX_DQ_WORKSPACE_BYTES:
        return None, "FlyDSL flex bwd requires a dQ workspace larger than 16 GiB"

    if (
        kv_num_blocks is None
        or kv_indices is None
        or full_kv_num_blocks is None
        or full_kv_indices is None
    ):
        return (
            None,
            "FlyDSL flex bwd requires complete forward BlockMask metadata",
        )
    metadata, reason = _get_block_mask_metadata(
        query,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        prefix="FlyDSL flex bwd ",
        static_reason="requires static BlockMask shapes",
    )
    if metadata is None:
        return None, reason
    count_shape = metadata.count_shape
    index_shape = metadata.index_shape
    full_count_shape = metadata.full_count_shape
    full_index_shape = metadata.full_index_shape
    expected_rows = sq // 128
    if len(count_shape) != 3 or count_shape[2] != expected_rows:
        return None, "FlyDSL flex bwd requires one BlockMask row per 128 Q rows"
    mask_batch, mask_heads, _ = count_shape
    if mask_batch not in (1, b):
        return None, "FlyDSL flex bwd BlockMask batch dimension must be 1 or B"
    if mask_heads not in (1, h):
        return None, "FlyDSL MHA bwd BlockMask head dimension must be 1 or H"
    if (
        len(index_shape) != 4
        or index_shape[:3] != count_shape
        or full_count_shape != count_shape
        or len(full_index_shape) != 4
        or full_index_shape[:3] != count_shape
        or index_shape[-1] <= 0
        or full_index_shape[-1] <= 0
    ):
        return None, "FlyDSL flex bwd received incompatible BlockMask metadata"
    max_partial_blocks = index_shape[-1]
    max_full_blocks = full_index_shape[-1]

    return (
        {
            "BATCH_SIZE": b,
            "NUM_HEADS": h,
            "SEQ_LEN": sq,
            "QK_HEAD_DIM": dqk,
            "V_HEAD_DIM": dv,
            "BLOCK_MASK_BATCH": mask_batch,
            "BLOCK_MASK_HEADS": mask_heads,
            "MAX_PARTIAL_BLOCKS": max_partial_blocks,
            "MAX_FULL_BLOCKS": max_full_blocks,
            "BLOCK_MASK_IS_DENSE": block_mask_is_dense,
            **mask_config,
            **common_config,
            "SM_SCALE": scale_value,
        },
        "",
    )


def _score_mod_capture_grad_mask(
    joint_graph: Subgraph | None, capture_count: int
) -> list[bool] | None:
    """Which captured score_mod tensors need gradients, or None if unknown."""
    if joint_graph is None:
        return [False] * capture_count
    outputs = [
        node for node in joint_graph.graph_module.graph.nodes if node.op == "output"
    ]
    if len(outputs) != 1:
        return None
    joint_outputs = outputs[0].args[0]
    if not isinstance(joint_outputs, (tuple, list)):
        return [False] * capture_count
    # Outputs are (score, b, h, q_idx, kv_idx, *captured buffer gradients).
    capture_outputs = list(joint_outputs[5:])
    if len(capture_outputs) > capture_count:
        return None
    capture_outputs += [None] * (capture_count - len(capture_outputs))
    return [output is not None for output in capture_outputs]


def _get_capture_grads(
    config: dict[str, Any], needs_grad: list[bool]
) -> tuple[list[FlyDSLCaptureGrad | None] | None, str]:
    """Classify every trainable score capture, or explain the Triton fallback."""
    program = FlyDSLScoreProgram(
        instructions=config["SCORE_PROGRAM"],
        output=config["SCORE_PROGRAM_OUTPUT"],
        buffer_shapes=config["SCORE_BUFFER_SHAPES"],
        buffer_strides=config["SCORE_BUFFER_STRIDES"],
    )
    captures: list[FlyDSLCaptureGrad | None] = []
    for buffer_index, trainable in enumerate(needs_grad):
        if not trainable:
            captures.append(None)
            continue
        capture, reason = classify_capture_gradient(
            program,
            buffer_index,
            batch=config["BATCH_SIZE"],
            heads=config["NUM_HEADS"],
            sequence_length=config["SEQ_LEN"],
        )
        if capture is None:
            return None, f"FlyDSL flex bwd {reason}"
        captures.append(capture)
    accumulated = [
        capture
        for capture in captures
        if capture is not None and capture.kind != CAPTURE_GRAD_UNUSED
    ]
    if len(accumulated) > _MAX_CAPTURE_GRADS:
        return None, (
            f"FlyDSL flex bwd supports at most {_MAX_CAPTURE_GRADS} trainable "
            "score_mod captures"
        )
    return captures, ""


def _capture_keeps(capture) -> tuple[bool, bool]:
    """Whether the capture is indexed by batch and by head."""
    names = [dim[0] for dim in capture.dims]
    return "b" in names, "h" in names


def _finish_capture_grad(capture, reduced, buffer, *, batch, heads, sequence):
    """Lay out a capture's reduced gradient in the buffer's shape and dtype.

    All sums ran in the kernel in a fixed order; this is pure data movement,
    so the gradient is bitwise reproducible across runs and recompilations.
    """
    shape = [V.graph.sizevars.guard_int(size) for size in buffer.get_size()]
    if capture.kind == CAPTURE_GRAD_UNUSED:
        gradient = full(shape, 0.0, dtype=torch.float32, device=buffer.get_device())
    elif capture.kind == CAPTURE_GRAD_FULL:
        gradient = view(reduced, shape)
    else:
        names = [dim[0] for dim in capture.dims]
        # Reduced layout is [B or 1, H or 1, columns]; drop collapsed axes.
        axes = [name for name in ("b", "h") if name in names]
        *leading, columns = capture_grad_reduction_shape(
            capture.kind, batch, heads, sequence, *_capture_keeps(capture)
        )
        sizes = [size for name, size in zip(("b", "h"), leading) if name in names]
        if capture.kind != CAPTURE_GRAD_INVARIANT:
            axes.append(
                {
                    CAPTURE_GRAD_KEY: "kv",
                    CAPTURE_GRAD_QUERY: "q",
                    CAPTURE_GRAD_RELATIVE: "rel",
                }[capture.kind]
            )
            sizes.append(columns)
        gradient = view(reduced, sizes)
        if capture.kind == CAPTURE_GRAD_RELATIVE:
            # Column c holds table entry c - (sequence - 1) + offset.
            extent = shape[names.index("rel")]
            start = sequence - 1 - capture.dims[names.index("rel")][2]
            pad_left = max(0, -start)
            pad_right = max(0, start + extent - (2 * sequence - 1))
            if pad_left or pad_right:
                gradient = constant_pad_nd(gradient, [pad_left, pad_right])
            start += pad_left
            gradient = slice_(gradient, len(axes) - 1, start, start + extent)
        gradient = permute(gradient, [axes.index(name) for name in names])
        gradient = view(gradient, shape)
    if buffer.get_dtype() != torch.float32:
        gradient = to_dtype(gradient, buffer.get_dtype())
    return gradient


def _empty_gradient(source, size, *, dtype, device):
    return empty_strided(
        size=size,
        stride=infer_dense_strides(size, source.get_stride()),
        dtype=dtype,
        device=device,
    )


def create_flydsl_flex_attention_backward_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    out: TensorBox,
    logsumexp: TensorBox,
    grad_out: TensorBox,
    grad_logsumexp: TensorBox | None,
    scale: float,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    fw_subgraph: Subgraph | None = None,
    mask_graph: Subgraph | None = None,
    joint_graph: Subgraph | None = None,
    score_mod_other_buffers: list[TensorBox] | None = None,
    mask_mod_other_buffers: list[TensorBox] | None = None,
    kv_num_blocks: TensorBox | None = None,
    kv_indices: TensorBox | None = None,
    full_kv_num_blocks: TensorBox | None = None,
    full_kv_indices: TensorBox | None = None,
    dq_accum_fp32: bool = True,
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple] | None:
    """Create a FlyDSL backward kernel, or return None for Triton fallback."""
    if not isinstance(dq_accum_fp32, bool):
        raise ValueError("FlyDSL FLYDSL_DQ_ACCUM_FP32 must be a bool")
    if not runtime_available():
        return None
    if fw_subgraph is None or mask_graph is None:
        raise AssertionError("FlyDSL backward requires the original mod graphs")
    capture_needs_grad = _score_mod_capture_grad_mask(
        joint_graph, len(score_mod_other_buffers or ())
    )
    if capture_needs_grad is None:
        return None

    block_mask_is_dense = (
        sparse_q_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and sparse_kv_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and is_trivial_mask_graph(mask_graph.graph_module)
    )
    if block_mask_is_dense:
        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ) = _create_dense_metadata(query, key)
        sparse_q_block_size = sparse_kv_block_size = 128

    (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
    )
    score_mod_other_buffers = maybe_realize(list(score_mod_other_buffers or []))
    mask_mod_other_buffers = maybe_realize(list(mask_mod_other_buffers or []))
    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    config, _ = _get_flydsl_flex_attention_backward_config(
        fw_subgraph,
        mask_graph,
        query,
        key=key,
        value=value,
        out=out,
        grad_out=grad_out,
        grad_logsumexp=grad_logsumexp,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        scale=scale,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        dq_accum_fp32=dq_accum_fp32,
        block_mask_is_dense=block_mask_is_dense,
    )
    if config is None:
        return None
    capture_grads: list[FlyDSLCaptureGrad | None] = [None] * len(
        score_mod_other_buffers
    )
    if any(capture_needs_grad):
        captures, _ = _get_capture_grads(config, capture_needs_grad)
        if captures is None:
            return None
        capture_grads = captures
        # Only the block-list pipeline accumulates capture gradients; dense
        # metadata is a valid (complete) block list.
        config["BLOCK_MASK_IS_DENSE"] = False
    if (
        kv_num_blocks is None
        or kv_indices is None
        or full_kv_num_blocks is None
        or full_kv_indices is None
    ):
        raise AssertionError("FlyDSL backward requires complete KV metadata")

    batch_size, num_heads, seq_len_q, head_dim = query.get_size()
    _, num_heads_kv, seq_len_kv, v_head_dim = value.get_size()
    device = query.get_device()
    dtype = query.get_dtype()
    if device is None:
        raise AssertionError("Device must not be None")

    grad_query, grad_key, grad_value = (
        _empty_gradient(source, size, dtype=dtype, device=device)
        for source, size in (
            (query, [batch_size, num_heads, seq_len_q, head_dim]),
            (key, [batch_size, num_heads_kv, seq_len_kv, head_dim]),
            (value, [batch_size, num_heads_kv, seq_len_kv, v_head_dim]),
        )
    )

    b = config["BATCH_SIZE"]
    h = config["NUM_HEADS"]
    s = config["SEQ_LEN"]
    q_chunks = s // _METADATA_BLOCK_SIZE
    kv_chunks = s // _METADATA_BLOCK_SIZE
    bh = b * h

    def make_scratch(numel: int, scratch_dtype: torch.dtype) -> TensorBox:
        return empty_strided(
            size=[numel],
            stride=[1],
            dtype=scratch_dtype,
            device=device,
        )

    delta = make_scratch(bh * s, torch.float32)
    traversal, _ = classify_mask_traversal(
        config["MASK_PROGRAM"],
        config["MASK_PROGRAM_OUTPUT"],
        config["MASK_BUFFER_SHAPES"],
        sequence_length=s,
        block_mask_is_dense=config["BLOCK_MASK_IS_DENSE"],
    )
    key_rows = 128 if traversal == MASK_TRAVERSAL_BLOCK_LIST else 192
    config["DQ_ACCUM_FP32"] = dq_accum_fp32
    workspace_dtype = torch.float32 if dq_accum_fp32 else torch.bfloat16
    workspace_element_bytes = 4 if dq_accum_fp32 else 2
    dq_partitions = choose_dq_partitions(
        bh, s, key_rows, workspace_element_bytes, qk_head_dim=config["QK_HEAD_DIM"]
    )
    grad_query_workspace = make_scratch(
        bh * dq_partitions * s * config["QK_HEAD_DIM"], workspace_dtype
    )
    if traversal != MASK_TRAVERSAL_BLOCK_LIST:
        kv_chunks = q_chunks = 1
    # Zero-initialized private workspaces, in kernel slot order.
    accumulated_captures = [
        (index, capture)
        for index, capture in enumerate(capture_grads)
        if capture is not None and capture.kind != CAPTURE_GRAD_UNUSED
    ]
    capture_workspaces = []
    capture_reduced = []
    for index, capture in accumulated_captures:
        workspace_shape = capture_grad_workspace_shape(
            capture.kind,
            bh,
            s,
            dq_partitions,
            [
                V.graph.sizevars.guard_int(size)
                for size in score_mod_other_buffers[index].get_size()
            ],
        )
        (workspace,) = maybe_realize(
            # 1-D so Inductor cannot pad the strides the kernel indexes densely.
            [
                full(
                    [math.prod(workspace_shape)],
                    0.0,
                    dtype=torch.float32,
                    device=device,
                )
            ]
        )
        capture_workspaces.append(workspace)
        # Full captures are written in place and leave their reduction unused.
        reduced_shape = (
            [1]
            if capture.kind == CAPTURE_GRAD_FULL
            else list(
                capture_grad_reduction_shape(
                    capture.kind, b, h, s, *_capture_keeps(capture)
                )
            )
        )
        capture_reduced.append(
            empty_strided(
                [math.prod(reduced_shape)], [1], dtype=torch.float32, device=device
            )
        )
    config["CAPTURE_GRADS"] = tuple(
        (
            capture.kind,
            capture.load_slot,
            capture.dims[[dim[0] for dim in capture.dims].index("rel")][1]
            if capture.kind == CAPTURE_GRAD_RELATIVE
            else 1,
        )
        for _, capture in accumulated_captures
    )
    config["CAPTURE_REDUCTIONS"] = tuple(
        _capture_keeps(capture) for _, capture in accumulated_captures
    )
    config["CAPTURE_GRAD_COUNT"] = len(accumulated_captures)
    (
        partial_kv_counts,
        partial_kv_indices,
        full_kv_counts,
        full_kv_indices_scratch,
    ) = (
        make_scratch(numel, torch.int32)
        for numel in (
            bh * kv_chunks,
            bh * kv_chunks * q_chunks,
            bh * kv_chunks,
            bh * kv_chunks * q_chunks,
        )
    )

    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=[sympy.sympify(s) for s in grad_query.get_stride()],
    )

    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
    config.update(
        {
            f"D{name}_STRIDE": tuple(
                V.graph.sizevars.guard_int(item) for item in grad.get_stride()
            )
            for name, grad in zip(
                ("Q", "K", "V"), (grad_query, grad_key, grad_value), strict=True
            )
        }
    )

    scratch_buffers = [
        delta,
        grad_query_workspace,
        partial_kv_counts,
        partial_kv_indices,
        full_kv_counts,
        full_kv_indices_scratch,
    ]
    input_nodes: list[TensorBox] = [
        query,
        key,
        value,
        out,
        grad_out,
        logsumexp,
        grad_key,
        grad_value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        *scratch_buffers,
    ]

    mask_buffer_count = config["MASK_BUFFER_COUNT"]
    if mask_buffer_count:
        if len(mask_mod_other_buffers or ()) != mask_buffer_count:
            raise AssertionError("mask_mod capture count changed during lowering")
        input_nodes.extend(mask_mod_other_buffers or ())
    score_buffer_count = config["SCORE_BUFFER_COUNT"]
    if score_buffer_count:
        if len(score_mod_other_buffers or ()) != score_buffer_count:
            raise AssertionError("score_mod capture count changed during lowering")
        input_nodes.extend(_kernel_capture_buffers(score_mod_other_buffers or ()))
    for workspace, reduced in zip(capture_workspaces, capture_reduced):
        input_nodes.extend((workspace, reduced))

    choices: list[Any] = []
    error = flex_flydsl_backward_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=output_layout,
        mutated_inputs=[
            grad_key,
            grad_value,
            *scratch_buffers,
            *capture_workspaces,
            *capture_reduced,
        ],
        SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
        SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
        **config,
    )
    if not choices:
        raise RuntimeError(f"FlyDSL template failed: {error}")

    template_output, _ = autotune_select_algorithm(
        "flex_flydsl_attention_backward",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns={
            8: create_num_blocks_fake_generator(kv_indices),
            9: create_indices_fake,
            10: create_num_blocks_fake_generator(full_kv_indices),
            11: create_indices_fake,
        },
        return_multi_template=False,
    )
    if not any(capture_needs_grad):
        return (template_output, grad_key, grad_value, tuple())
    results = {
        index: workspace if capture.kind == CAPTURE_GRAD_FULL else reduced
        for (index, capture), workspace, reduced in zip(
            accumulated_captures, capture_workspaces, capture_reduced
        )
    }
    captured_grads = tuple(
        None
        if capture is None
        else _finish_capture_grad(
            capture,
            results.get(index),
            score_mod_other_buffers[index],
            batch=b,
            heads=h,
            sequence=s,
        )
        for index, capture in enumerate(capture_grads)
    )
    return (template_output, grad_key, grad_value, captured_grads)
