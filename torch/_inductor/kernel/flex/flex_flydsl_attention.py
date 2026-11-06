# mypy: allow-untyped-defs

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE

from ...codegen.flydsl import flydsl_utils
from ...codegen.flydsl.flydsl_template import FlyDSLTemplate
from ...ir import FixedLayout, Pointwise
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

_MAX_BUFFER_BYTES = 1 << 32


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


@dataclass(frozen=True)
class _BlockMaskMetadata:
    count_shape: tuple[int, ...]
    index_shape: tuple[int, ...]
    full_count_shape: tuple[int, ...]
    full_index_shape: tuple[int, ...]
    full_numel: int


def _get_block_mask_metadata(
    query,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    *,
    prefix: str = "",
    static_reason: str = "requires statically known BlockMask dimensions",
) -> tuple[_BlockMaskMetadata | None, str]:
    nodes = (
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    )
    try:
        shapes = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_size())
            for node in nodes
        )
        full_numel = V.graph.sizevars.guard_int(full_kv_num_blocks.get_numel())
        dtypes = tuple(node.get_dtype() for node in nodes)
        devices = tuple(node.get_device() for node in nodes)
    except (AttributeError, TypeError, ValueError):
        return None, f"{prefix}{static_reason}"

    if any(dtype != torch.int32 for dtype in dtypes):
        return None, f"{prefix}requires int32 BlockMask metadata"
    if any(device != query.get_device() for device in devices):
        return None, f"{prefix}requires BlockMask metadata on the query device"
    try:
        strides = tuple(
            tuple(V.graph.sizevars.guard_int(item) for item in node.get_stride())
            for node in nodes
        )
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None, f"{prefix}requires statically known BlockMask metadata strides"
    if not all(
        _is_contiguous_shape_stride(shape, stride)
        for shape, stride in zip(shapes, strides)
    ):
        return None, f"{prefix}requires contiguous BlockMask metadata"
    return _BlockMaskMetadata(
        shapes[0], shapes[1], shapes[2], shapes[3], full_numel
    ), ""


def _get_mask_program_config(mask_graph, mask_mod_other_buffers):
    if is_trivial_mask_graph(mask_graph.graph_module):
        return {
            "MASK_PROGRAM": (),
            "MASK_PROGRAM_OUTPUT": 0,
            "MASK_BUFFER_COUNT": 0,
            "MASK_BUFFER_SHAPES": (),
            "MASK_BUFFER_STRIDES": (),
        }, ""
    mask_program, reason = lower_flydsl_mask_graph(
        mask_graph.graph_module,
        mask_mod_other_buffers,
    )
    if mask_program is None:
        return None, f"unsupported mask_mod: {reason}"
    return {
        "MASK_PROGRAM": mask_program.instructions,
        "MASK_PROGRAM_OUTPUT": mask_program.output,
        "MASK_BUFFER_COUNT": mask_program.buffer_count,
        "MASK_BUFFER_SHAPES": mask_program.buffer_shapes,
        "MASK_BUFFER_STRIDES": mask_program.buffer_strides,
    }, ""


def _is_gfx950_device(device) -> bool:
    if not flydsl_utils.runtime_available() or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")
    return str(arch).split(":", 1)[0] == "gfx950"


def _get_flydsl_common_config(
    *,
    query,
    key,
    value,
    subgraph,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    extra_strides=(),
    allow_mask_mod_buffers: bool = False,
    allow_strided_bhsd: bool = False,
) -> tuple[dict[str, tuple[int, ...] | None] | None, str]:
    device = query.get_device()
    if device is None or device.type != "cuda" or not _is_gfx950_device(device):
        return None, "requires ROCm gfx950 and the FlyDSL runtime"
    if query.get_dtype() != torch.bfloat16:
        return None, f"supports BF16 only, got {query.get_dtype()}"
    if query.get_dtype() != key.get_dtype() or query.get_dtype() != value.get_dtype():
        return None, "requires query, key, and value to have the same dtype"
    if not is_trivial_score_graph(subgraph.graph_module):
        return None, "supports identity score_mod only"
    if score_mod_other_buffers:
        return None, "does not support captured score_mod buffers"
    if mask_mod_other_buffers and not allow_mask_mod_buffers:
        return None, "does not support captured mask_mod buffers"

    named_tensors = (
        ("Q_STRIDE", query),
        ("K_STRIDE", key),
        ("V_STRIDE", value),
        *extra_strides,
    )
    strides = tuple(
        None
        if node is None
        else _get_supported_bhsd_stride(node, allow_strided=allow_strided_bhsd)
        for _, node in named_tensors
    )
    if any(
        node is not None and stride is None
        for (_, node), stride in zip(named_tensors, strides, strict=True)
    ):
        layout = "4D BHSD tensors with contiguous head dimensions"
        if not allow_strided_bhsd:
            layout = "contiguous 4D BHSD tensors"
        return None, f"requires {layout}"
    if not all(
        _fits_u32_head_slice(node) for _, node in named_tensors if node is not None
    ):
        return None, "requires every per-head tensor slice to be smaller than 4 GiB"

    return {
        name: stride for (name, _), stride in zip(named_tensors, strides, strict=True)
    }, ""


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
        sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
        sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
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

    metadata, reason = _get_block_mask_metadata(
        query,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        static_reason="requires statically known tensor and BlockMask dimensions",
    )
    if metadata is None:
        return None, reason
    mask_shape = metadata.count_shape
    index_shape = metadata.index_shape
    full_count_shape = metadata.full_count_shape
    full_index_shape = metadata.full_index_shape

    mask_config, reason = _get_mask_program_config(mask_graph, mask_mod_other_buffers)
    if mask_config is None:
        return None, reason

    decode = 0 < sq < 128
    common_config, reason = _get_flydsl_common_config(
        query=query,
        key=key,
        value=value,
        subgraph=subgraph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        allow_mask_mod_buffers=bool(mask_config["MASK_BUFFER_COUNT"]),
        allow_strided_bhsd=not decode,
    )
    if common_config is None:
        return None, reason

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

    has_full_blocks = metadata.full_numel != 0
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
            **mask_config,
            **common_config,
            "SM_SCALE": scale_value,
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
