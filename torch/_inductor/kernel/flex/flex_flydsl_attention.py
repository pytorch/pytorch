# mypy: allow-untyped-defs

import operator
from collections.abc import Callable, Sequence
from typing import Any

import sympy

import torch

from ...codegen.flydsl import flydsl_utils
from ...codegen.flydsl.flydsl_template import FlyDSLTemplate
from ...codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    runtime_available,
)
from ...ir import FixedLayout, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import V
from .common import (
    create_indices_fake,
    create_num_blocks_fake_generator,
    infer_dense_strides,
    load_flex_template,
)
from .flex_flash_attention import is_trivial_mask_graph, is_trivial_score_graph
from .flex_flydsl_mask import lower_flydsl_mask_graph


flex_flydsl_backward_template = FlyDSLTemplate(
    name="flex_flydsl_backward", source=load_flex_template("flydsl_backward")
)

_MAX_BUFFER_BYTES = 1 << 32
_BWD_Q_CHUNK_SIZE = 64
_BWD_KV_CHUNK_SIZE = 64
_ADD_TARGETS = (
    operator.add,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
)
_GE_TARGETS = (operator.ge, torch.ops.aten.ge.Tensor, torch.ops.aten.ge.Scalar)
_LE_TARGETS = (operator.le, torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar)


def is_causal_mask_graph(graph_module, q_offset: int = 0) -> bool:
    nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    if len(placeholders) != 4 or len(outputs) != 1:
        return False
    result = outputs[0].args[0]
    if not hasattr(result, "target") or len(result.args) != 2:
        return False
    lhs, rhs = result.args
    query = placeholders[2]

    def is_offset_query(value) -> bool:
        if q_offset == 0 and value is query:
            return True
        if (
            not hasattr(value, "target")
            or value.target not in _ADD_TARGETS
            or len(value.args) < 2
        ):
            return False
        add_lhs, add_rhs = value.args[:2]
        return (
            add_lhs is query
            and isinstance(add_rhs, (int, float))
            and add_rhs == q_offset
        ) or (
            add_rhs is query
            and isinstance(add_lhs, (int, float))
            and add_lhs == q_offset
        )

    return (
        result.target in _GE_TARGETS and is_offset_query(lhs) and rhs is placeholders[3]
    ) or (
        result.target in _LE_TARGETS and lhs is placeholders[3] and is_offset_query(rhs)
    )


def _get_supported_bhsd_stride(node, *, allow_strided: bool) -> tuple[int, ...] | None:
    try:
        sizes = [V.graph.sizevars.guard_int(value) for value in node.get_size()]
        strides = [V.graph.sizevars.guard_int(value) for value in node.get_stride()]
    except (TypeError, ValueError):
        return None
    if len(sizes) != 4 or len(strides) != 4:
        return None
    contiguous = [
        sizes[1] * sizes[2] * sizes[3],
        sizes[2] * sizes[3],
        sizes[3],
        1,
    ]
    if strides == contiguous:
        return tuple(strides)
    if allow_strided and strides[-1] == 1 and all(stride > 0 for stride in strides):
        return tuple(strides)
    return None


def _fits_u32_buffer(node) -> bool:
    try:
        sizes = [V.graph.sizevars.guard_int(value) for value in node.get_size()]
        strides = [V.graph.sizevars.guard_int(value) for value in node.get_stride()]
        element_size = torch._utils._element_size(node.get_dtype())
    except (AttributeError, TypeError, ValueError):
        return False
    if len(sizes) != len(strides) or any(stride < 0 for stride in strides):
        return False
    storage_elements = 1 + sum(
        (size - 1) * stride for size, stride in zip(sizes, strides)
    )
    return storage_elements * element_size < _MAX_BUFFER_BYTES


def _is_gfx950_device(device) -> bool:
    if not flydsl_utils.runtime_available() or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")
    return str(arch).split(":", 1)[0] == "gfx950"


def _check_flydsl_common_compatibility(
    *,
    query,
    key,
    value,
    subgraph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    extra_tensors=(),
    allow_mask_mod_buffers: bool = False,
    allow_strided_bhsd: bool = False,
) -> str:
    device = query.get_device()
    if device is None or device.type != "cuda" or not _is_gfx950_device(device):
        return "requires ROCm gfx950 and the FlyDSL runtime"
    if query.get_dtype() != torch.bfloat16:
        return f"supports BF16 only, got {query.get_dtype()}"
    if query.get_dtype() != key.get_dtype() or query.get_dtype() != value.get_dtype():
        return "requires query, key, and value to have the same dtype"
    if not is_trivial_score_graph(subgraph.graph_module):
        return "supports identity score_mod only"
    if score_mod_other_buffers or (
        mask_mod_other_buffers and not allow_mask_mod_buffers
    ):
        return "does not support captured score_mod or mask_mod buffers"

    tensors = (query, key, value, *extra_tensors)
    if not all(
        _get_supported_bhsd_stride(node, allow_strided=allow_strided_bhsd) is not None
        for node in tensors
        if node is not None
    ):
        layout = "4D BHSD tensors with contiguous head dimensions"
        if not allow_strided_bhsd:
            layout = "contiguous 4D BHSD tensors"
        return f"requires {layout}"
    if not all(_fits_u32_buffer(node) for node in tensors if node is not None):
        return "requires every tensor buffer to be smaller than 4 GiB"
    return ""


def _flydsl_unavailable_message() -> str:
    reason = _flydsl_runtime_unavailable_reason()
    if reason is None:
        reason = "FlyDSL runtime is unavailable"
    return (
        f"FlyDSL flex attention backward is unavailable: {reason}. "
        "It requires ROCm/gfx950 and the optional `flydsl` runtime (0.3.x)."
    )


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
) -> tuple[dict[str, Any] | None, str]:
    score_mod_other_buffers = score_mod_other_buffers or ()
    mask_mod_other_buffers = mask_mod_other_buffers or ()

    if not runtime_available():
        return None, _flydsl_unavailable_message()

    if torch.version.hip is None:
        return None, "FlyDSL flex bwd requires ROCm/gfx950"

    device = query.get_device() if hasattr(query, "get_device") else None
    if device is not None and not _is_gfx950_device(device):
        return None, "FlyDSL flex bwd requires ROCm gfx950"

    if query.get_dtype() != torch.bfloat16:
        return (
            None,
            f"FlyDSL flex bwd supports bf16 only, got {query.get_dtype()}",
        )

    if not is_trivial_score_graph(fw_subgraph.graph_module):
        return None, "FlyDSL flex bwd supports identity score_mod only"

    if score_mod_other_buffers:
        return None, "FlyDSL flex bwd does not support captured score_mod buffers"

    trivial_mask = is_trivial_mask_graph(mask_graph.graph_module)
    causal_mask = is_causal_mask_graph(mask_graph.graph_module)
    mask_program = None
    if not trivial_mask and not causal_mask:
        mask_program, mask_reason = lower_flydsl_mask_graph(
            mask_graph.graph_module,
            mask_mod_other_buffers,
        )
        if mask_program is None:
            return None, f"unsupported mask_mod: {mask_reason}"

    common_reason = _check_flydsl_common_compatibility(
        query=query,
        key=key,
        value=value,
        subgraph=fw_subgraph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        extra_tensors=(out, grad_out),
        allow_mask_mod_buffers=mask_program is not None,
        allow_strided_bhsd=True,
    )
    if common_reason:
        return None, f"FlyDSL flex bwd {common_reason}"

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
    try:
        count_shape = [
            V.graph.sizevars.guard_int(item) for item in kv_num_blocks.get_size()
        ]
        index_shape = [
            V.graph.sizevars.guard_int(item) for item in kv_indices.get_size()
        ]
        full_count_shape = [
            V.graph.sizevars.guard_int(item) for item in full_kv_num_blocks.get_size()
        ]
        full_index_shape = [
            V.graph.sizevars.guard_int(item) for item in full_kv_indices.get_size()
        ]
    except (AttributeError, TypeError, ValueError):
        return None, "FlyDSL flex bwd requires static BlockMask shapes"
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

    q_stride = _get_supported_bhsd_stride(query, allow_strided=True)
    k_stride = _get_supported_bhsd_stride(key, allow_strided=True)
    v_stride = _get_supported_bhsd_stride(value, allow_strided=True)
    out_stride = (
        None if out is None else _get_supported_bhsd_stride(out, allow_strided=True)
    )
    grad_out_stride = (
        None
        if grad_out is None
        else _get_supported_bhsd_stride(grad_out, allow_strided=True)
    )
    if q_stride is None or k_stride is None or v_stride is None:
        return None, "FlyDSL flex bwd requires supported Q/K/V BHSD strides"
    if out is not None and out_stride is None:
        return None, "FlyDSL flex bwd requires supported OUT BHSD strides"
    if grad_out is not None and grad_out_stride is None:
        return None, "FlyDSL flex bwd requires supported grad_out BHSD strides"

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
            "CAUSAL_PARTIAL_BLOCKS": causal_mask,
            "MASK_PROGRAM": (() if mask_program is None else mask_program.instructions),
            "MASK_PROGRAM_OUTPUT": (0 if mask_program is None else mask_program.output),
            "MASK_BUFFER_COUNT": (
                0 if mask_program is None else mask_program.buffer_count
            ),
            "MASK_BUFFER_SHAPES": (
                () if mask_program is None else mask_program.buffer_shapes
            ),
            "MASK_BUFFER_STRIDES": (
                () if mask_program is None else mask_program.buffer_strides
            ),
            "Q_STRIDE": q_stride,
            "K_STRIDE": k_stride,
            "V_STRIDE": v_stride,
            "OUT_STRIDE": out_stride,
            "DO_STRIDE": grad_out_stride,
            "SM_SCALE": scale_value,
        },
        "",
    )


def _can_use_flydsl_flex_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    query: TensorBox,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
    **kwargs: Any,
) -> tuple[bool, str]:
    config, reason = _get_flydsl_flex_attention_backward_config(
        fw_subgraph,
        mask_graph,
        query,
        score_mod_other_buffers,
        **kwargs,
    )
    return config is not None, reason


def _use_flydsl_flex_attention_backward(
    fw_subgraph: Subgraph,
    mask_graph: Subgraph,
    backend: str,
    query: TensorBox,
    score_mod_other_buffers: Sequence[TensorBox] | None = None,
    **kwargs: Any,
) -> bool:
    """Determine if we should use FlyDSL flex attention backward.

    FlyDSL is experimental and must be explicitly requested via BACKEND='FLYDSL'.
    Mirrors ``_use_flex_flash_attention_backward``: raises when the backend is
    requested but cannot be satisfied.
    """
    if backend != "FLYDSL":
        return False

    can_use, reason = _can_use_flydsl_flex_attention_backward(
        fw_subgraph,
        mask_graph,
        query,
        score_mod_other_buffers,
        **kwargs,
    )

    if not can_use:
        raise RuntimeError(
            f"BACKEND='FLYDSL' but FlyDSL flex backward cannot be used: {reason}"
        )

    return True


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
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple]:
    """Create a FlyDSL flex attention backward kernel for supported inputs."""
    if not runtime_available():
        raise RuntimeError(_flydsl_unavailable_message())
    if fw_subgraph is None or mask_graph is None:
        raise AssertionError("FlyDSL backward requires the original mod graphs")

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
    )
    if config is None:
        raise RuntimeError(f"FlyDSL flex backward cannot be used: {reason}")
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

    grad_query_strides = infer_dense_strides(
        [batch_size, num_heads, seq_len_q, head_dim], query.get_stride()
    )
    grad_query = empty_strided(
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=grad_query_strides,
        dtype=dtype,
        device=device,
    )

    grad_key_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, head_dim], key.get_stride()
    )
    grad_key = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, head_dim],
        stride=grad_key_strides,
        dtype=dtype,
        device=device,
    )

    grad_value_strides = infer_dense_strides(
        [batch_size, num_heads_kv, seq_len_kv, v_head_dim], value.get_stride()
    )
    grad_value = empty_strided(
        size=[batch_size, num_heads_kv, seq_len_kv, v_head_dim],
        stride=grad_value_strides,
        dtype=dtype,
        device=device,
    )

    b = config["BATCH_SIZE"]
    h = config["NUM_HEADS"]
    s = config["SEQ_LEN"]
    q_chunks = s // _BWD_Q_CHUNK_SIZE
    kv_chunks = s // _BWD_KV_CHUNK_SIZE
    bh = b * h

    def make_scratch(numel: int, scratch_dtype: torch.dtype) -> TensorBox:
        return empty_strided(
            size=[numel],
            stride=[1],
            dtype=scratch_dtype,
            device=device,
        )

    delta = make_scratch(bh * s, torch.float32)
    partial_q_counts = make_scratch(bh * q_chunks, torch.int32)
    partial_q_indices = make_scratch(bh * q_chunks * kv_chunks, torch.int32)
    full_q_counts = make_scratch(bh * q_chunks, torch.int32)
    full_q_indices_scratch = make_scratch(bh * q_chunks * kv_chunks, torch.int32)
    partial_kv_counts = make_scratch(bh * kv_chunks, torch.int32)
    partial_kv_indices = make_scratch(bh * kv_chunks * q_chunks, torch.int32)
    full_kv_counts = make_scratch(bh * kv_chunks, torch.int32)
    full_kv_indices_scratch = make_scratch(bh * kv_chunks * q_chunks, torch.int32)

    # we use dq as the output layout
    output_layout = FixedLayout(
        device=device,
        dtype=dtype,
        size=[batch_size, num_heads, seq_len_q, head_dim],
        stride=[sympy.sympify(s) for s in grad_query.get_stride()],
    )

    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
    config["DQ_STRIDE"] = tuple(
        V.graph.sizevars.guard_int(item) for item in grad_query.get_stride()
    )
    config["DK_STRIDE"] = tuple(
        V.graph.sizevars.guard_int(item) for item in grad_key.get_stride()
    )
    config["DV_STRIDE"] = tuple(
        V.graph.sizevars.guard_int(item) for item in grad_value.get_stride()
    )

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
        delta,
        partial_q_counts,
        partial_q_indices,
        full_q_counts,
        full_q_indices_scratch,
        partial_kv_counts,
        partial_kv_indices,
        full_kv_counts,
        full_kv_indices_scratch,
    ]

    mask_buffer_count = config["MASK_BUFFER_COUNT"]
    if mask_buffer_count:
        if len(mask_mod_other_buffers or ()) != mask_buffer_count:
            raise AssertionError("mask_mod capture count changed during lowering")
        input_nodes.extend(mask_mod_other_buffers or ())

    choices: list[Any] = []
    error = flex_flydsl_backward_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=output_layout,
        mutated_inputs=[
            grad_key,
            grad_value,
            delta,
            partial_q_counts,
            partial_q_indices,
            full_q_counts,
            full_q_indices_scratch,
            partial_kv_counts,
            partial_kv_indices,
            full_kv_counts,
            full_kv_indices_scratch,
        ],
        SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
        SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
        **config,
    )

    if not choices:
        raise RuntimeError(f"FlyDSL template failed: {error}")

    input_gen_fns: dict[int, Callable] = {
        8: create_num_blocks_fake_generator(kv_indices),
        9: create_indices_fake,
        10: create_num_blocks_fake_generator(full_kv_indices),
        11: create_indices_fake,
    }

    template_output, _ = autotune_select_algorithm(
        "flex_flydsl_attention_backward",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_multi_template=False,
    )

    return (template_output, grad_key, grad_value, tuple())
