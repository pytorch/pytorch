# mypy: allow-untyped-defs

from collections.abc import Callable, Sequence
from typing import Any

import sympy

import torch
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE

from ...codegen.flydsl import flydsl_utils
from ...codegen.flydsl.flydsl_template import FlyDSLTemplate
from ...codegen.flydsl.flydsl_utils import (
    _flydsl_runtime_unavailable_reason,
    runtime_available,
)
from ...ir import FixedLayout, Pointwise, ShapeAsConstantBuffer, Subgraph, TensorBox
from ...lowering import empty_strided, full
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import ops, V
from .common import (
    construct_strides,
    create_indices_fake,
    create_num_blocks_fake_generator,
    freeze_irnodes,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    load_flex_template,
    maybe_realize,
)
from .flex_flash_attention import is_trivial_mask_graph, is_trivial_score_graph
from .flex_flydsl_mask import lower_flydsl_mask_graph


flex_flydsl_forward_template = FlyDSLTemplate(
    name="flex_flydsl_forward",
    source=load_flex_template("flydsl_forward"),
)

flex_flydsl_backward_template = FlyDSLTemplate(
    name="flex_flydsl_backward", source=load_flex_template("flydsl_backward")
)

_MAX_BUFFER_BYTES = 1 << 32
_BWD_METADATA_BLOCK_SIZE = 128


def _contiguous_strides(shape):
    return construct_strides(shape, range(len(shape) - 1, -1, -1))


def _get_supported_bhsd_stride(node, *, allow_strided: bool) -> tuple[int, ...] | None:
    try:
        sizes = [V.graph.sizevars.guard_int(value) for value in node.get_size()]
        strides = [V.graph.sizevars.guard_int(value) for value in node.get_stride()]
    except (TypeError, ValueError):
        return None
    if len(sizes) != 4 or len(strides) != 4:
        return None
    if strides == _contiguous_strides(sizes):
        return tuple(strides)
    # A zero stride is not a broadcast when the corresponding extent is one.
    if (
        allow_strided
        and strides[-1] == 1
        and all(
            stride > 0 or (size == 1 and stride == 0)
            for size, stride in zip(sizes, strides)
        )
    ):
        return tuple(strides)
    return None


def _fits_u32_head_slice(node) -> bool:
    try:
        sizes = [V.graph.sizevars.guard_int(value) for value in node.get_size()]
        strides = [V.graph.sizevars.guard_int(value) for value in node.get_stride()]
        element_size = torch._utils._element_size(node.get_dtype())
    except (AttributeError, TypeError, ValueError):
        return False
    if len(sizes) != 4 or len(strides) != 4 or any(stride < 0 for stride in strides):
        return False
    storage_elements = 1 + sum(
        (size - 1) * stride for size, stride in zip(sizes[-2:], strides[-2:])
    )
    return storage_elements * element_size < _MAX_BUFFER_BYTES


def _is_contiguous_shape_stride(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> bool:
    if len(shape) != len(stride):
        return False
    expected = _contiguous_strides(tuple(max(size, 1) for size in shape))
    return all(
        size == 1 or actual == contiguous
        for size, actual, contiguous in zip(shape, stride, expected)
    )


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
    if score_mod_other_buffers:
        return "does not support captured score_mod buffers"
    if mask_mod_other_buffers and not allow_mask_mod_buffers:
        return "does not support captured mask_mod buffers"

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
    if not all(_fits_u32_head_slice(node) for node in tensors if node is not None):
        return "requires every per-head tensor slice to be smaller than 4 GiB"
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
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None, "requires statically known BlockMask metadata strides"
    if not all(
        _is_contiguous_shape_stride(shape, stride)
        for shape, stride in zip(metadata_shapes, metadata_strides)
    ):
        return None, "requires contiguous BlockMask metadata"

    trivial_mask = is_trivial_mask_graph(mask_graph.graph_module)
    mask_program = None
    if not trivial_mask:
        mask_program, mask_reason = lower_flydsl_mask_graph(
            mask_graph.graph_module,
            mask_mod_other_buffers,
        )
        if mask_program is None:
            return None, f"unsupported mask_mod: {mask_reason}"

    decode = 0 < sq < 128
    common_reason = _check_flydsl_common_compatibility(
        query=query,
        key=key,
        value=value,
        subgraph=subgraph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        allow_mask_mod_buffers=mask_program is not None,
        allow_strided_bhsd=not decode,
    )
    if common_reason:
        return None, common_reason

    q_stride = _get_supported_bhsd_stride(query, allow_strided=not decode)
    k_stride = _get_supported_bhsd_stride(key, allow_strided=not decode)
    v_stride = _get_supported_bhsd_stride(value, allow_strided=not decode)
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

    gqa_group_size = hq // hkv
    packed_decode_rows = gqa_group_size * sq
    supports_prefill = sq % 128 == 0 and mask_shape[2] == sq // 128
    supports_decode = (
        decode
        and mask_shape[1] in (1, hkv)
        and mask_shape[2] == 1
        and 0 < packed_decode_rows <= 256
    )
    if sk % 128 != 0:
        return None, "requires Sk divisible by 128"
    if sparse_q_block_size != 128 or sparse_kv_block_size != 128:
        return None, "requires sparse Q/KV block sizes of 128"
    if not has_full_blocks or max_full_blocks <= 0 or index_shape[-1] <= 0:
        return None, "requires non-empty partial and full BlockMask storage"
    if not (supports_prefill or supports_decode):
        return (
            None,
            "requires prefill Sq divisible by 128 with matching BlockMask rows, "
            "or decode 0 < Sq < 128 with a shared/per-KV-head BlockMask and "
            "(Hq/Hkv)*Sq <= 256",
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
    write_max_scores=True,
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

    config["WRITE_MAX_SCORES"] = bool(write_max_scores)

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


def _create_dense_metadata(query, key):
    """Materialize only the forward metadata for the frontend's no-mask sentinel."""
    seq_q = V.graph.sizevars.guard_int(query.get_size()[2])
    seq_kv = V.graph.sizevars.guard_int(key.get_size()[2])
    num_q_blocks = (seq_q + 127) // 128
    num_kv_blocks = (seq_kv + 127) // 128
    shape = [1, 1, num_q_blocks]
    device = query.get_device()
    return (
        full(shape, 0, dtype=torch.int32, device=device),
        full([*shape, 1], 0, dtype=torch.int32, device=device),
        full(shape, num_kv_blocks, dtype=torch.int32, device=device),
        Pointwise.create(
            device=device,
            dtype=torch.int32,
            ranges=[*shape, num_kv_blocks],
            inner_fn=lambda index: ops.index_expr(index[-1], torch.int32),
        ),
    )


def create_flydsl_flex_attention_kernel(
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
    subgraph_buffer,
    mask_graph_buffer,
    write_max_scores=True,
):
    """Lower the explicitly selected FlyDSL backend independently of Triton."""
    if (
        sparse_q_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and sparse_kv_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and is_trivial_mask_graph(mask_graph.graph_module)
    ):
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
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
    )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)
    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    batch, heads, seq_q, _ = query.get_size()
    out_size = [batch, heads, seq_q, value.get_size()[-1]]
    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        out_size,
        stride=infer_dense_strides(out_size, query.get_stride()),
    )
    logsumexp = empty_strided(
        [batch, heads, seq_q], None, dtype=torch.float32, device=query.get_device()
    )
    max_scores = empty_strided(
        [batch, heads, seq_q], None, dtype=torch.float32, device=query.get_device()
    )
    choices = []
    appended, reason = maybe_append_flydsl_flex_attention_choice(
        choices,
        query=query,
        key=key,
        value=value,
        logsumexp=logsumexp,
        max_scores=max_scores,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        layout=layout,
        subgraph=subgraph,
        mask_graph=mask_graph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        scale=scale,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        write_max_scores=write_max_scores,
    )
    if not appended:
        raise RuntimeError(
            "BACKEND='FLYDSL' but the FlyDSL flex forward candidate "
            f"could not be registered: {reason}"
        )
    out, _ = autotune_select_algorithm(
        "flex_attention_flydsl",
        choices,
        [
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            *mask_mod_other_buffers,
        ],
        layout,
        input_gen_fns={
            5: create_num_blocks_fake_generator(kv_indices),
            6: create_indices_fake,
            7: create_num_blocks_fake_generator(full_kv_indices),
            8: create_indices_fake,
        },
    )
    out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    out.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )
    return out, logsumexp, max_scores


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
    mask_program = None
    if not trivial_mask:
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
        metadata_nodes = (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        )
        metadata_shapes = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_size())
            for node in metadata_nodes
        )
        metadata_dtypes = tuple(node.get_dtype() for node in metadata_nodes)
        metadata_devices = tuple(node.get_device() for node in metadata_nodes)
    except (AttributeError, TypeError, ValueError):
        return None, "FlyDSL flex bwd requires static BlockMask shapes"
    if any(dtype != torch.int32 for dtype in metadata_dtypes):
        return None, "FlyDSL flex bwd requires int32 BlockMask metadata"
    if any(device != query.get_device() for device in metadata_devices):
        return None, "FlyDSL flex bwd requires BlockMask metadata on the query device"
    try:
        metadata_strides = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_stride())
            for node in metadata_nodes
        )
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return (
            None,
            "FlyDSL flex bwd requires statically known BlockMask metadata strides",
        )
    if not all(
        _is_contiguous_shape_stride(shape, stride)
        for shape, stride in zip(metadata_shapes, metadata_strides)
    ):
        return None, "FlyDSL flex bwd requires contiguous BlockMask metadata"
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
) -> tuple[TensorBox | ShapeAsConstantBuffer, TensorBox, TensorBox, tuple]:
    """Create a FlyDSL flex attention backward kernel for supported inputs."""
    if not isinstance(dq_accum_fp32, bool):
        raise ValueError("FlyDSL FLYDSL_DQ_ACCUM_FP32 must be a bool")
    if not runtime_available():
        raise RuntimeError(_flydsl_unavailable_message())
    if fw_subgraph is None or mask_graph is None:
        raise AssertionError("FlyDSL backward requires the original mod graphs")

    if (
        sparse_q_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and sparse_kv_block_size == _LARGE_SPARSE_BLOCK_SIZE
        and is_trivial_mask_graph(mask_graph.graph_module)
    ):
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
    q_chunks = s // _BWD_METADATA_BLOCK_SIZE
    kv_chunks = s // _BWD_METADATA_BLOCK_SIZE
    bh = b * h

    def make_scratch(numel: int, scratch_dtype: torch.dtype) -> TensorBox:
        return empty_strided(
            size=[numel],
            stride=[1],
            dtype=scratch_dtype,
            device=device,
        )

    delta = make_scratch(bh * s, torch.float32)
    # Dense ranges need no reverse lists. Allocate minimal placeholders for
    # the common ABI; sparse traversal uses one list per 128-row KV owner.
    from ..vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
        choose_dq_partitions,
        classify_mask_traversal,
        MASK_TRAVERSAL_BLOCK_LIST,
    )

    traversal, _ = classify_mask_traversal(
        config["MASK_PROGRAM"],
        config["MASK_PROGRAM_OUTPUT"],
        config["MASK_BUFFER_SHAPES"],
        sequence_length=s,
    )
    key_rows = 128 if traversal == MASK_TRAVERSAL_BLOCK_LIST else 192
    config["DQ_ACCUM_FP32"] = dq_accum_fp32
    workspace_dtype = torch.float32 if dq_accum_fp32 else torch.bfloat16
    workspace_element_bytes = 4 if dq_accum_fp32 else 2
    dq_partitions = choose_dq_partitions(bh, s, key_rows, workspace_element_bytes)
    grad_query_workspace = make_scratch(
        bh * dq_partitions * s * config["QK_HEAD_DIM"], workspace_dtype
    )
    if traversal != MASK_TRAVERSAL_BLOCK_LIST:
        kv_chunks = q_chunks = 1
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
        grad_query_workspace,
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
            grad_query_workspace,
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
