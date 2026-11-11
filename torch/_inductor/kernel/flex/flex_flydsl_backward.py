# mypy: allow-untyped-defs

import logging
from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._dynamo.utils import counters
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE

from ...codegen.flydsl.flydsl_flex_kernel import FlyDSLFlexTemplate
from ...codegen.flydsl.flydsl_utils import runtime_available
from ...ir import (
    ExternKernel,
    FixedLayout,
    Pointwise,
    ShapeAsConstantBuffer,
    Subgraph,
    TensorBox,
)
from ...lowering import constant_pad_nd, empty_strided, lowerings, to_dtype, view
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import ops, V
from ..vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
    backward_key_rows,
    backward_single_pass,
    choose_dq_partitions,
    classify_mask_traversal,
    dq_workspace_head_group,
    MASK_TRAVERSAL_BLOCK_LIST,
    MAX_BUFFER_VIEW_BYTES,
    MAX_DQ_WORKSPACE_BYTES,
    MAX_REVERSE_LIST_QUERY_BLOCKS,
)
from ..vendored_templates.flydsl.kernels.flex_attn_plan import backward_plan
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    infer_dense_strides,
    load_flex_template,
    maybe_realize,
)
from .flex_flash_attention import is_trivial_mask_graph
from .flex_flydsl_attention import (
    _bhsd_sizes,
    _BlockStates,
    _broadcast_kv_batch,
    _capture_table_reason,
    _ceil_div,
    _create_dense_metadata,
    _get_block_mask_metadata,
    _get_flydsl_common_config,
    _get_mask_program_config,
    _get_score_mod_config,
    _head_dim_reason,
    _is_runtime,
    _maybe_split_block_mask,
    _mod_inputs,
    _pad_head_dim,
    _prepare_captures,
    _require_bhsd_stride,
    _size_expr,
    _sparse_block_size_reason,
    _sym,
    _syms,
    _target_limits,
    _template_int,
    _template_strides,
    _unpad_head_dim,
    kernel_head_dims,
)


flex_flydsl_backward_template = FlyDSLFlexTemplate(
    name="flex_flydsl_backward", source=load_flex_template("flydsl_backward")
)

aten = torch.ops.aten
log = logging.getLogger(__name__)

_METADATA_BLOCK_SIZE = 128
# Inductor's sum over a short leading dim reads the partials uncoalesced; an
# unrolled fp32 add is several times faster up to this many query batches.
_MAX_UNROLLED_BATCH_SUM = 32

perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")


def _record_backward_fallback(reason: str) -> None:
    counters["inductor"]["flydsl_flex_bwd_fallback"] += 1
    perf_hint_log.info(
        "FlyDSL flex attention backward falling back to Triton: %s", reason
    )


def _padded_query_length(seq_len):
    # Delta and the dQ workspace cover whole BlockMask rows.
    return _ceil_div(seq_len, _METADATA_BLOCK_SIZE) * _METADATA_BLOCK_SIZE


def _hint(value):
    return value if isinstance(value, int) else value.node.hint


def _dq_partitions(
    *,
    batch,
    heads,
    kv_heads,
    seq_q,
    seq_kv,
    dims,
    block_list,
    element_bytes,
    captures,
    target,
):
    """The dQ slot count, chosen at the size hints. With runtime sizes it is
    kept for every size whose workspace still fits; raises ValueError if the
    workspace does not fit."""
    qk_dim, v_dim = dims
    partitions = choose_dq_partitions(
        _hint(batch) * heads,
        _padded_query_length(_hint(seq_q)),
        backward_key_rows(block_list, qk_dim, v_dim, captures),
        element_bytes,
        qk_head_dim=qk_dim,
        key_sequence_length=_hint(seq_kv),
        compute_batch_heads=_hint(batch) * kv_heads,
        block_list=block_list,
        target=target,
    )
    slot_bytes = _padded_query_length(seq_q) * qk_dim * element_bytes * partitions
    head_group = dq_workspace_head_group(
        batch * heads if isinstance(batch, int) else heads
    )
    if (
        batch * heads * slot_bytes > MAX_DQ_WORKSPACE_BYTES
        or head_group * slot_bytes > MAX_BUFFER_VIEW_BYTES
    ):
        raise ValueError("FlyDSL backward dQ workspace exceeds its buffer limits")
    return partitions


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
    joint_outputs: Any = None,
    block_states: _BlockStates | None = None,
    single_pass: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """With ``single_pass``, take the backward of ``backward_single_pass`` dims."""
    score_mod_other_buffers = score_mod_other_buffers or ()
    mask_mod_other_buffers = mask_mod_other_buffers or ()

    mask_config, reason = _get_mask_program_config(
        mask_graph, mask_mod_other_buffers, block_states, query
    )
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
        allow_score_mod=joint_outputs is not None,
    )
    if common_config is None:
        return None, f"FlyDSL flex bwd {reason}"
    score_mod_config, _, reason = _get_score_mod_config(
        fw_subgraph, score_mod_other_buffers, query, mask_config["MASK_BUFFER_COUNT"]
    )
    if score_mod_config is None:
        return None, f"FlyDSL flex bwd unsupported score_mod: {reason}"
    if score_mod_config["HAS_SCORE_MOD"]:
        if joint_outputs is None:
            return None, "FlyDSL flex bwd score_mod requires its joint graph"
        if joint_outputs.captured_grads_compute or joint_outputs.mutated_grads:
            return (
                None,
                "FlyDSL flex bwd does not support gradients for captured tensors",
            )

    if scale is None:
        return None, "FlyDSL flex bwd requires static shapes and scale"

    try:
        (b, h, sq, dqk), (bk, hk, sk, dk), (bv, hv, sv, dv) = (
            _bhsd_sizes(node) for node in (query, key, value)
        )
        out_shape, grad_out_shape, grad_logsumexp_shape = (
            None if node is None else list(_syms(node.get_size()))
            for node in (out, grad_out, grad_logsumexp)
        )
        block_m = V.graph.sizevars.guard_int(sparse_q_block_size)
        block_n = V.graph.sizevars.guard_int(sparse_kv_block_size)
        scale_value = float(scale)
    except (AttributeError, TypeError, ValueError):
        return None, "FlyDSL flex bwd requires backed shapes and a static scale"

    if bv != bk or bk not in (1, b):
        return None, "FlyDSL flex bwd requires a K/V batch of 1 or the Q batch"
    _broadcast_kv_batch(common_config, b, bk)
    if dk != dqk:
        return None, "FlyDSL flex bwd requires matching Q/K head dims"
    if (hv, sv) != (hk, sk):
        return None, "FlyDSL flex bwd requires K and V to share heads and length"
    if hk <= 0 or h % hk:
        return None, "FlyDSL flex bwd requires Q heads divisible by KV heads"
    reason = _head_dim_reason(dqk, dv)
    if reason:
        return None, f"FlyDSL flex bwd {reason}"
    expected_out_shape = [b, h, sq, dv]
    if out_shape is not None and out_shape != expected_out_shape:
        return None, "FlyDSL flex bwd requires OUT shape [B, Hq, Sq, Dv]"
    if grad_out_shape is not None and grad_out_shape != expected_out_shape:
        return None, "FlyDSL flex bwd requires grad_out shape [B, Hq, Sq, Dv]"
    if out is not None and out.get_dtype() != query.get_dtype():
        return None, "FlyDSL flex bwd requires OUT in the query dtype"
    if grad_out is not None and grad_out.get_dtype() != query.get_dtype():
        return None, "FlyDSL flex bwd requires grad_out in the query dtype"
    reason = _sparse_block_size_reason(block_m, block_n)
    if reason:
        return None, f"FlyDSL flex bwd {reason}"
    if sq <= 0 or sk <= 0:
        return None, "FlyDSL flex bwd requires positive Q and KV sequence lengths"
    if grad_logsumexp_shape is not None and grad_logsumexp_shape != [b, h, sq]:
        return None, "FlyDSL flex bwd requires grad_logsumexp shape [B, Hq, Sq]"
    workspace_element_bytes = 4 if dq_accum_fp32 else 2
    padded_sq = _padded_query_length(sq)
    if b * h * padded_sq * dqk * workspace_element_bytes > MAX_DQ_WORKSPACE_BYTES:
        return None, "FlyDSL flex bwd requires a dQ workspace larger than 16 GiB"
    traversal, _ = classify_mask_traversal(
        mask_config["MASK_PROGRAM"],
        mask_config["MASK_PROGRAM_OUTPUT"],
        mask_config["MASK_BUFFER_SHAPES"],
        sequence_length=_template_int(sq),
        key_sequence_length=_template_int(sk),
        block_mask_is_dense=block_mask_is_dense,
    )
    target = _target_limits(query)
    try:
        block_list = traversal == MASK_TRAVERSAL_BLOCK_LIST
        _dq_partitions(
            batch=b,
            heads=h,
            kv_heads=hk,
            seq_q=sq,
            seq_kv=sk,
            dims=(dqk, dv),
            block_list=block_list,
            element_bytes=workspace_element_bytes,
            captures=mask_config["MASK_BUFFER_COUNT"]
            + score_mod_config.get("SCORE_MOD_BUFFER_COUNT", 0),
            target=target,
        )
    except ValueError:
        return (
            None,
            "FlyDSL flex bwd requires each dQ workspace view to fit a 32-bit "
            "buffer range",
        )
    if (
        traversal == MASK_TRAVERSAL_BLOCK_LIST
        and padded_sq // 128 > MAX_REVERSE_LIST_QUERY_BLOCKS
    ):
        return (
            None,
            "FlyDSL flex bwd BlockMask traversal supports at most "
            f"{MAX_REVERSE_LIST_QUERY_BLOCKS} query blocks",
        )

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
    if len(count_shape) != 3 or count_shape[2] != padded_sq // 128:
        return None, "FlyDSL flex bwd requires one BlockMask row per 128 Q rows"
    mask_batch, mask_heads, _ = count_shape
    if mask_batch not in (1, b):
        return None, "FlyDSL flex bwd BlockMask batch dimension must be 1 or B"
    if mask_heads not in (1, hk, h):
        return (
            None,
            "FlyDSL flex bwd BlockMask head dimension must be 1, KV heads, or Q heads",
        )
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
    sizes = {
        "BATCH_SIZE": _template_int(b),
        "SEQ_LEN_Q": _template_int(sq),
        "SEQ_LEN_KV": _template_int(sk),
        "BLOCK_MASK_BATCH": _template_int(mask_batch),
        "MAX_PARTIAL_BLOCKS": _template_int(index_shape[-1]),
        "MAX_FULL_BLOCKS": _template_int(full_index_shape[-1]),
    }
    reason = _capture_table_reason(mask_config, score_mod_config["CAPTURE_TABLE"])
    if reason:
        return None, f"FlyDSL flex bwd {reason}"
    single_pass = single_pass and backward_single_pass(dqk, dv, target)
    # Strides stay ints or SymInts until the gradient strides are known.
    return (
        {
            **sizes,
            "NUM_HEADS": h,
            "NUM_KV_HEADS": hk,
            "QK_HEAD_DIM": dqk,
            "V_HEAD_DIM": dv,
            "BLOCK_MASK_HEADS": mask_heads,
            "BLOCK_MASK_IS_DENSE": block_mask_is_dense,
            "HAS_DLSE": grad_logsumexp is not None,
            **({"SINGLE_PASS": True} if single_pass else {}),
            **mask_config,
            **score_mod_config,
            **common_config,
            "SM_SCALE": scale_value,
            "TARGET_LIMITS": tuple(target),
        },
        "",
    )


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
    score_mod_other_buffers: list[TensorBox] | None = None,
    mask_mod_other_buffers: list[TensorBox] | None = None,
    kv_num_blocks: TensorBox | None = None,
    kv_indices: TensorBox | None = None,
    full_kv_num_blocks: TensorBox | None = None,
    full_kv_indices: TensorBox | None = None,
    dq_accum_fp32: bool = True,
    fw_subgraph_buffer: Any = None,
    joint_outputs: Any = None,
    single_pass: bool = False,
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple] | None:
    """Create a FlyDSL backward kernel, or return None for Triton fallback."""
    for name, option in (
        ("FLYDSL_DQ_ACCUM_FP32", dq_accum_fp32),
        ("FLYDSL_WIDE_SINGLE_PASS", single_pass),
    ):
        if not isinstance(option, bool):
            raise ValueError(f"FlyDSL {name} must be a bool")
    if not runtime_available():
        _record_backward_fallback("FlyDSL runtime is unavailable")
        return None
    if fw_subgraph is None or mask_graph is None:
        raise AssertionError("FlyDSL backward requires the original mod graphs")

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
    gradient_sources = (query, key, value)
    qk_dim, v_dim = (
        V.graph.sizevars.guard_int(node.get_size()[-1]) for node in (query, value)
    )
    padded_dims = kernel_head_dims(qk_dim, v_dim)
    if padded_dims is not None:
        query, key = (_pad_head_dim(node, padded_dims[0]) for node in (query, key))
        value, out, grad_out = (
            _pad_head_dim(node, padded_dims[1]) for node in (value, out, grad_out)
        )
    query, out, grad_out = (
        _require_bhsd_stride(node) for node in (query, out, grad_out)
    )
    key, value = (
        _require_bhsd_stride(node, broadcast_batch=True) for node in (key, value)
    )
    *metadata, sparse_q_block_size, sparse_kv_block_size, block_states = (
        _maybe_split_block_mask(
            query,
            key,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            sparse_q_block_size,
            sparse_kv_block_size,
        )
    )
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = metadata
    score_mod_other_buffers, mask_mod_other_buffers, block_states = _prepare_captures(
        score_mod_other_buffers,
        mask_mod_other_buffers,
        block_states,
        query.get_device(),
    )

    config, reason = _get_flydsl_flex_attention_backward_config(
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
        joint_outputs=joint_outputs,
        block_states=block_states,
        single_pass=single_pass,
    )
    if config is None:
        _record_backward_fallback(reason)
        return None
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

    # With one K/V batch, dK/dV hold one partial per query batch until summed.
    grad_query, grad_key, grad_value = (
        _empty_gradient(source, size, dtype=dtype, device=device)
        for source, size in zip(
            gradient_sources,
            (
                [batch_size, num_heads, seq_len_q, head_dim],
                [batch_size, num_heads_kv, seq_len_kv, head_dim],
                [batch_size, num_heads_kv, seq_len_kv, v_head_dim],
            ),
            strict=True,
        )
    )

    b, h, sq, _ = _bhsd_sizes(query)
    sk = _sym(seq_len_kv)
    batch_expr, seq_q_expr, seq_kv_expr = map(
        _size_expr, (batch_size, seq_len_q, seq_len_kv)
    )
    s = _padded_query_length(seq_q_expr)
    q_chunks = _ceil_div(seq_q_expr, _METADATA_BLOCK_SIZE)
    kv_chunks = _ceil_div(seq_kv_expr, _METADATA_BLOCK_SIZE)
    bh = batch_expr * h

    def make_scratch(numel: int, scratch_dtype: torch.dtype) -> TensorBox:
        return empty_strided(
            size=[numel],
            stride=[1],
            dtype=scratch_dtype,
            device=device,
        )

    if grad_logsumexp is None:
        delta = make_scratch(bh * s, torch.float32)
    else:
        # The delta kernel overwrites its LSE-gradient input in the padded
        # delta layout, so pass a copy with zeros for rows past Sq.
        delta = constant_pad_nd(
            to_dtype(grad_logsumexp, torch.float32, copy=True), [0, s - seq_q_expr]
        )
        delta = view(delta, [bh * s])
        (delta,) = maybe_realize([ExternKernel.require_contiguous(delta)])
    traversal, _ = classify_mask_traversal(
        config["MASK_PROGRAM"],
        config["MASK_PROGRAM_OUTPUT"],
        config["MASK_BUFFER_SHAPES"],
        sequence_length=config["SEQ_LEN_Q"],
        key_sequence_length=config["SEQ_LEN_KV"],
        block_mask_is_dense=config["BLOCK_MASK_IS_DENSE"],
    )
    config["DQ_ACCUM_FP32"] = dq_accum_fp32
    workspace_dtype = torch.float32 if dq_accum_fp32 else dtype
    workspace_element_bytes = 4 if dq_accum_fp32 else 2
    block_list = traversal == MASK_TRAVERSAL_BLOCK_LIST
    dims = (config["QK_HEAD_DIM"], config["V_HEAD_DIM"])
    captures = config["MASK_BUFFER_COUNT"] + config.get("SCORE_MOD_BUFFER_COUNT", 0)
    dq_partitions = _dq_partitions(
        batch=b,
        heads=h,
        kv_heads=config["NUM_KV_HEADS"],
        seq_q=sq,
        seq_kv=sk,
        dims=dims,
        block_list=block_list,
        element_bytes=workspace_element_bytes,
        captures=captures,
        target=config["TARGET_LIMITS"],
    )
    grad_query_workspace = make_scratch(
        bh * dq_partitions * s * config["QK_HEAD_DIM"], workspace_dtype
    )
    if traversal != MASK_TRAVERSAL_BLOCK_LIST:
        kv_chunks = q_chunks = 1
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
            f"D{name}_STRIDE": _syms(grad.get_stride())
            for name, grad in zip(
                ("Q", "K", "V"), (grad_query, grad_key, grad_value), strict=True
            )
        }
    )
    stride_lengths = {
        "Q_STRIDE": sq,
        "K_STRIDE": sk,
        "V_STRIDE": sk,
        "OUT_STRIDE": sq,
        "DO_STRIDE": sq,
        "DQ_STRIDE": sq,
        "DK_STRIDE": sk,
        "DV_STRIDE": sk,
    }
    plan = backward_plan(
        batch_size=b,
        num_heads=h,
        num_kv_heads=config["NUM_KV_HEADS"],
        sequence_length=sq,
        kv_sequence_length=sk,
        qk_head_dim=dims[0],
        value_head_dim=dims[1],
        mask_traversal=traversal,
        workspace_element_bytes=workspace_element_bytes,
        strides=tuple(config[name] for name in stride_lengths),
        dq_partitions=dq_partitions,
        captures=captures,
        target=config["TARGET_LIMITS"],
    )
    for name, length in stride_lengths.items():
        config[name] = _template_strides(config[name], length)
    runtime = _is_runtime(
        [config[name] for name in ("BATCH_SIZE", "SEQ_LEN_Q", "SEQ_LEN_KV")]
        + [config[name] for name in stride_lengths]
        + [config["BLOCK_MASK_BATCH"], config["MAX_PARTIAL_BLOCKS"]]
        + [config["MAX_FULL_BLOCKS"], config["MASK_BUFFER_SHAPES"]]
    )
    config["RUNTIME_SIZES"] = runtime
    if runtime:
        config["PLAN"] = plan

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

    mod_inputs, mod_kwargs, reason = _mod_inputs(
        config,
        query,
        fw_subgraph,
        None if joint_outputs is None else fw_subgraph_buffer,
        score_mod_other_buffers,
        mask_mod_other_buffers,
        block_states,
        None,
    )
    if mod_inputs is None:
        raise AssertionError(f"FlyDSL backward {reason}")
    input_nodes.extend(mod_inputs)
    if mod_kwargs:
        mod_kwargs["subgraphs"].append(joint_outputs.grad_input)
    mutated_inputs = [grad_key, grad_value, *scratch_buffers]
    if config["CAPTURE_TABLE"]:
        capture_table = make_scratch(
            config["MASK_BUFFER_COUNT"] + config.get("SCORE_MOD_BUFFER_COUNT", 0),
            torch.int64,
        )
        input_nodes.append(capture_table)
        mutated_inputs.append(capture_table)

    choices: list[Any] = []
    error = flex_flydsl_backward_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=output_layout,
        mutated_inputs=mutated_inputs,
        **mod_kwargs,
        SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
        SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
        **config,
    )
    if not choices:
        if config["HAS_SCORE_MOD"]:
            _record_backward_fallback(f"score_mod template failed: {error}")
            return None
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
    widths = (qk_dim, qk_dim, v_dim)
    template_output = _unpad_gradient(template_output, gradient_sources[0], widths[0])
    if _sym(key.get_size()[0]) != b:
        grad_key, grad_value = (
            _sum_query_batches(_unpad_head_dim(grad, width), b)
            for grad, width in zip((grad_key, grad_value), widths[1:])
        )
    else:
        grad_key, grad_value = (
            _unpad_gradient(grad, source, width)
            for grad, source, width in zip(
                (grad_key, grad_value), gradient_sources[1:], widths[1:]
            )
        )
    return (template_output, grad_key, grad_value, tuple())


def _unpad_gradient(grad, source, width):
    size = [*grad.get_size()[:-1], width]
    return _unpad_head_dim(grad, width, infer_dense_strides(size, source.get_stride()))


def _sum_query_batches(grad, batch_size):
    """Reduce per-query-batch dK/dV partials to the single K/V batch."""
    if not isinstance(batch_size, int) or batch_size > _MAX_UNROLLED_BATCH_SUM:
        return lowerings[aten.sum](grad, axis=0, keepdims=True)
    load = grad.make_loader()

    def inner_fn(index):
        _, *rest = index
        total = ops.to_dtype(load([0, *rest]), torch.float32)
        for batch in range(1, batch_size):
            total = ops.add(total, ops.to_dtype(load([batch, *rest]), torch.float32))
        return ops.to_dtype(total, grad.get_dtype())

    return Pointwise.create(
        device=grad.get_device(),
        dtype=grad.get_dtype(),
        inner_fn=inner_fn,
        ranges=[1, *grad.get_size()[1:]],
    )
