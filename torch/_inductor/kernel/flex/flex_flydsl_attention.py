# mypy: allow-untyped-defs

import operator
from typing import Any

import torch

from ...codegen.flydsl import flydsl_utils
from ...codegen.flydsl.flydsl_template import FlyDSLTemplate
from ...virtualized import V
from .common import infer_dense_strides, load_flex_template
from .flex_flash_attention import is_trivial_mask_graph, is_trivial_score_graph
from .flex_flydsl_mask import lower_flydsl_mask_graph


flex_flydsl_forward_template = FlyDSLTemplate(
    name="flex_flydsl_forward",
    source=load_flex_template("flydsl_forward"),
)

_MAX_BUFFER_BYTES = 1 << 32
_ADD_TARGETS = (
    operator.add,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
)
_ALPHA_ADD_TARGETS = (
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
        if value.target in _ALPHA_ADD_TARGETS and value.kwargs.get("alpha", 1) != 1:
            return False
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


def _is_contiguous_shape_stride(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> bool:
    if len(shape) != len(stride):
        return False
    expected_stride = 1
    for size, actual_stride in reversed(tuple(zip(shape, stride))):
        if size != 1 and actual_stride != expected_stride:
            return False
        expected_stride *= max(size, 1)
    return True


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


def _get_flydsl_flex_attention_forward_config(
    *,
    query,
    key,
    value,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    subgraph,
    mask_graph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    scale,
    sparse_q_block_size,
    sparse_kv_block_size,
) -> tuple[dict[str, Any] | None, str]:
    if full_kv_num_blocks is None or full_kv_indices is None:
        return None, "requires full_kv_num_blocks/full_kv_indices metadata"

    metadata_nodes = (
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    )
    try:
        b, hq, sq, qk_dim = [
            V.graph.sizevars.guard_int(item) for item in query.get_size()
        ]
        bkv, hkv, sk, key_dim = [
            V.graph.sizevars.guard_int(item) for item in key.get_size()
        ]
        bv, hv, sv, v_dim = [
            V.graph.sizevars.guard_int(item) for item in value.get_size()
        ]
        mask_shape = [
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
        metadata_shapes = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_size())
            for node in metadata_nodes
        )
        metadata_dtypes = tuple(node.get_dtype() for node in metadata_nodes)
        metadata_devices = tuple(node.get_device() for node in metadata_nodes)
        sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
        sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
        full_numel = V.graph.sizevars.guard_int(full_kv_num_blocks.get_numel())
        scale_value = float(scale)
        output_stride = tuple(
            V.graph.sizevars.guard_int(item)
            for item in infer_dense_strides(
                [b, hq, sq, v_dim],
                query.get_stride(),
            )
        )
    except (AttributeError, TypeError, ValueError):
        return None, "requires statically known tensor and BlockMask dimensions"

    if any(dtype != torch.int32 for dtype in metadata_dtypes):
        return None, "requires int32 BlockMask metadata"
    if any(device != query.get_device() for device in metadata_devices):
        return None, "requires BlockMask metadata on the query device"
    try:
        metadata_strides = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_stride())
            for node in metadata_nodes
        )
    except NotImplementedError:
        # Pointwise-created metadata is realized before template registration,
        # where this check runs again with a concrete layout.
        metadata_strides = None
    except (AttributeError, TypeError, ValueError):
        return None, "requires statically known BlockMask metadata strides"
    if metadata_strides is not None:
        if not all(
            _is_contiguous_shape_stride(shape, stride)
            for shape, stride in zip(metadata_shapes, metadata_strides)
        ):
            return None, "requires contiguous BlockMask metadata"

    causal_mask = is_causal_mask_graph(
        mask_graph.graph_module,
        sk - sq,
    )
    trivial_mask = is_trivial_mask_graph(mask_graph.graph_module)
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
        subgraph=subgraph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        allow_mask_mod_buffers=mask_program is not None,
        allow_strided_bhsd=True,
    )
    if common_reason:
        return None, common_reason

    q_stride = _get_supported_bhsd_stride(query, allow_strided=True)
    k_stride = _get_supported_bhsd_stride(key, allow_strided=True)
    v_stride = _get_supported_bhsd_stride(value, allow_strided=True)
    if q_stride is None or k_stride is None or v_stride is None:
        return None, "requires supported Q/K/V BHSD strides"

    if (bkv, hkv, sk) != (bv, hv, sv):
        return None, "requires key and value to have matching B/Hkv/Sk dimensions"
    if b != bkv:
        return None, "does not yet support broadcasted K/V batches"
    if qk_dim != key_dim:
        return None, "requires query and key to have the same head dimension"
    if (qk_dim, v_dim) not in ((128, 128), (192, 128)):
        return (
            None,
            "supports only (QK head dim, V head dim) = (128, 128) or (192, 128)",
        )
    if hkv <= 0 or hq % hkv != 0 or sq <= 0 or sk <= 0:
        return None, "requires positive lengths and Hq divisible by Hkv"
    if len(mask_shape) != 3 or len(index_shape) != 4:
        return None, "requires 3D BlockMask counts and 4D BlockMask indices"
    if index_shape[:3] != mask_shape:
        return None, "requires matching BlockMask count/index leading dimensions"
    if mask_shape[0] not in (1, b):
        return None, "BlockMask batch dimension must be 1 or B"
    if mask_shape[1] not in (1, hkv, hq):
        return None, "BlockMask head dimension must be 1, Hkv, or Hq"
    if sparse_q_block_size <= 0 or sparse_kv_block_size <= 0:
        return None, "requires positive sparse block sizes"

    has_full_blocks = full_numel != 0
    max_full_blocks = 1
    if has_full_blocks:
        if full_count_shape != mask_shape:
            return None, "requires matching partial/full BlockMask count dimensions"
        if len(full_index_shape) != 4 or full_index_shape[:3] != mask_shape:
            return None, "requires matching full BlockMask count/index dimensions"
        max_full_blocks = full_index_shape[-1]

    candidate_blocks = 2 * (max_full_blocks + index_shape[-1])
    supports_prefill = (
        sq % 256 == 0 and candidate_blocks <= 512 and mask_shape[2] == sq // 128
    )
    if sk % 128 != 0:
        return None, "requires Sk divisible by 128"
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        return None, "requires sparse Q/KV block sizes of 128"
    if not has_full_blocks or max_full_blocks <= 0 or index_shape[-1] <= 0:
        return None, "requires non-empty partial and full BlockMask storage"
    if not supports_prefill:
        return (
            None,
            "requires prefill Sq divisible by 256 with matching BlockMask rows "
            "and at most 512 stored candidate blocks per CTA",
        )

    return (
        {
            "BATCH_SIZE": b,
            "NUM_Q_HEADS": hq,
            "NUM_KV_HEADS": hkv,
            "SEQ_Q": sq,
            "SEQ_KV": sk,
            "QK_HEAD_DIM": qk_dim,
            "V_HEAD_DIM": v_dim,
            "BLOCK_MASK_BATCH": mask_shape[0],
            "BLOCK_MASK_HEADS": mask_shape[1],
            "NUM_Q_BLOCKS": mask_shape[2],
            "MAX_PARTIAL_BLOCKS": index_shape[-1],
            "MAX_FULL_BLOCKS": max_full_blocks,
            "SPARSE_Q_BLOCK_SIZE": sparse_q_block_size,
            "SPARSE_KV_BLOCK_SIZE": sparse_kv_block_size,
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
            "SM_SCALE": scale_value,
            "Q_STRIDE": q_stride,
            "K_STRIDE": k_stride,
            "V_STRIDE": v_stride,
            "O_STRIDE": output_stride,
        },
        "",
    )


def can_use_flydsl_flex_attention_forward(
    *,
    query,
    key,
    value,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    subgraph,
    mask_graph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    scale,
    sparse_q_block_size,
    sparse_kv_block_size,
) -> tuple[bool, str]:
    config, reason = _get_flydsl_flex_attention_forward_config(
        query=query,
        key=key,
        value=value,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        subgraph=subgraph,
        mask_graph=mask_graph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        scale=scale,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
    )
    return config is not None, reason


def maybe_append_flydsl_flex_attention_choice(
    choices,
    *,
    query,
    key,
    value,
    logsumexp,
    max_scores,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    layout,
    subgraph,
    mask_graph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    scale,
    sparse_q_block_size,
    sparse_kv_block_size,
) -> tuple[bool, str]:
    config, reason = _get_flydsl_flex_attention_forward_config(
        query=query,
        key=key,
        value=value,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        subgraph=subgraph,
        mask_graph=mask_graph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        scale=scale,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
    )
    if config is None:
        return False, reason

    input_nodes = [
        query,
        key,
        value,
        logsumexp,
        max_scores,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ]
    mask_buffer_count = config["MASK_BUFFER_COUNT"]
    if mask_buffer_count:
        if len(mask_mod_other_buffers) != mask_buffer_count:
            return False, "mask_mod capture count changed during lowering"
        input_nodes.extend(mask_mod_other_buffers)

    choices_before = len(choices)
    error = flex_flydsl_forward_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        mutated_inputs=[logsumexp, max_scores],
        layout=layout,
        **config,
    )
    if len(choices) == choices_before:
        return False, f"FlyDSL template registration failed: {error}"
    return True, ""
