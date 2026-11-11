# mypy: allow-untyped-defs

import functools
import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import sympy

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing

from ...codegen.flydsl import flydsl_utils
from ...codegen.flydsl.flydsl_flex_kernel import FlyDSLFlexTemplate
from ...ir import ExternKernel, FixedLayout, Pointwise, Reduction, TensorBox
from ...lowering import constant_pad_nd, empty_strided, full, lowerings
from ...ops_handler import ReductionType
from ...select_algorithm import autotune_select_algorithm
from ...virtualized import ops, V
from ..vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
    forward_single_pass,
    forward_value_slice,
    MAX_DIRECT_CAPTURES,
    SUPPORTED_HEAD_DIMS,
    WIDE_HEAD_DIMS,
)
from ..vendored_templates.flydsl.kernels.flex_attn_plan import (
    forward_kv_splits,
    forward_plan,
    use_head_slow_order,
)
from ..vendored_templates.flydsl.kernels.flex_attn_target import (
    FLEX_ARCH,
    target_limits,
    TargetLimits,
    WAVE_SIZE,
)
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
from .flex_flydsl_mask import (
    add_block_states,
    BLOCK_STATE_EMPTY,
    BLOCK_STATE_FULL,
    BLOCK_STATE_PARTIAL,
    lower_flydsl_mask_graph,
    mask_capture_tensors,
    prepare_flydsl_mask_captures,
)


log = logging.getLogger(__name__)

aten = torch.ops.aten

flex_flydsl_forward_template = FlyDSLFlexTemplate(
    name="flex_flydsl_forward",
    source=load_flex_template("flydsl_forward"),
)
flex_flydsl_split_kv_combine_template = FlyDSLFlexTemplate(
    name="flex_flydsl_split_kv_combine",
    source=load_flex_template("flydsl_split_kv_combine"),
)

_MAX_BUFFER_BYTES = 1 << 32
_FLYDSL_DTYPES = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_SCORE_MOD_CAPTURE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_MAX_HEAD_DIM = max(max(dims) for dims in WIDE_HEAD_DIMS)


def _contiguous_strides(shape):
    return construct_strides(shape, range(len(shape) - 1, -1, -1))


def _sym(expr):
    """``expr`` as an int, or as a SymInt whose comparisons add shape guards."""
    if isinstance(expr, (int, sympy.Integer)):
        return int(expr)
    expr = V.graph.sizevars.simplify(expr)
    if isinstance(expr, sympy.Integer):
        return int(expr)
    if free_unbacked_symbols(expr):
        raise ValueError(f"FlyDSL flex attention needs backed sizes, got {expr}")
    return V.graph.sizevars.shape_env.create_symintnode(expr, hint=None)


def _syms(values):
    return tuple(_sym(value) for value in values)


def _expr(value):
    """The sympy expression of an int or SymInt size."""
    return value.node.expr if isinstance(value, torch.SymInt) else sympy.Integer(value)


def _size_expr(expr):
    """An int for a static size, else its sympy expression."""
    if isinstance(expr, (int, sympy.Integer)):
        return int(expr)
    expr = V.graph.sizevars.simplify(expr)
    return int(expr) if isinstance(expr, sympy.Integer) else expr


def _max_size(left, right):
    if isinstance(left, int) and isinstance(right, int):
        return max(left, right)
    return V.graph.sizevars.evaluate_max(left, right)


def _ceil_div(value, divisor):
    """Ceil division of an int, SymInt or size expression."""
    if isinstance(value, (int, torch.SymInt)):
        return -(-value // divisor)
    return _size_expr(CeilDiv(value, divisor))


def _template_int(value):
    """A size for the kernel builders: the int, or None to read it at runtime."""
    return value if isinstance(value, int) else None


def _template_strides(strides, sequence_length):
    """Strides for the kernel builders: ints, ``("seq", c)`` for ``c`` times the
    runtime ``sequence_length``, or None to read at runtime."""
    encoded: list[int | tuple[str, int] | None] = []
    for stride in strides:
        if isinstance(stride, int):
            encoded.append(stride)
            continue
        ratio = V.graph.sizevars.simplify(_expr(stride) / _expr(sequence_length))
        encoded.append(
            ("seq", int(ratio)) if isinstance(ratio, sympy.Integer) else None
        )
    return tuple(encoded)


def _is_runtime(value) -> bool:
    if isinstance(value, (tuple, list)):
        return any(_is_runtime(item) for item in value)
    return value is None


def _get_supported_bhsd_stride(node, *, broadcast_batch: bool = False) -> tuple | None:
    """The strides of a 4D tensor the kernels read in place, as ints or SymInts."""
    try:
        sizes = list(_syms(node.get_size()))
        strides = list(_syms(node.get_stride()))
    except (TypeError, ValueError):
        return None
    if len(sizes) != 4 or len(strides) != 4:
        return None
    if strides == _contiguous_strides(sizes):
        return tuple(strides)
    # A zero stride is not a broadcast when the corresponding extent is one.
    if strides[-1] == 1 and all(
        stride > 0 or (stride == 0 and (size == 1 or (dim == 0 and broadcast_batch)))
        for dim, (size, stride) in enumerate(zip(sizes, strides))
    ):
        return tuple(strides)
    return None


def _require_bhsd_stride(node, *, broadcast_batch: bool = False):
    """Copy a 4D tensor the kernels cannot read in place, such as one whose head
    dim is not unit-strided, to dense strides that keep its B/H/S order."""
    if _get_supported_bhsd_stride(node, broadcast_batch=broadcast_batch) is not None:
        return node
    try:
        sizes, strides = node.get_size(), node.get_stride()
    except (AttributeError, NotImplementedError):
        return node
    if len(sizes) != 4 or len(strides) != 4:
        return node
    return ExternKernel.require_exact_strides(node, infer_dense_strides(sizes, strides))


def kernel_head_dims(qk_dim: int, v_dim: int) -> tuple[int, int] | None:
    """The supported (QK, V) head dims with the fewest zero-padded columns that
    cover ``(qk_dim, v_dim)``; dims above 256 only if none up to 256 does."""
    for table in (SUPPORTED_HEAD_DIMS, WIDE_HEAD_DIMS):
        covering = [dims for dims in table if dims[0] >= qk_dim and dims[1] >= v_dim]
        if covering:
            return min(covering, key=lambda dims: (dims[0] + dims[1], dims[0]))
    return None


def _head_dim_reason(qk_dim: int, v_dim: int) -> str:
    if (qk_dim, v_dim) in SUPPORTED_HEAD_DIMS + WIDE_HEAD_DIMS:
        return ""
    if kernel_head_dims(qk_dim, v_dim) is None:
        return (
            f"supports QK and V head dims up to {_MAX_HEAD_DIM}, got "
            f"({qk_dim}, {v_dim})"
        )
    return (
        "supports only (QK head dim, V head dim) in "
        f"{SUPPORTED_HEAD_DIMS + WIDE_HEAD_DIMS}; the lowering pads narrower "
        "heads to one of them"
    )


def _pad_head_dim(node, width: int):
    """Zero-pad the head dim of a BHSD tensor to ``width`` in a dense copy that
    keeps its B/H/S order and a zero batch stride."""
    sizes, strides = list(node.get_size()), list(node.get_stride())
    head_dim = V.graph.sizevars.guard_int(sizes[-1])
    if head_dim == width:
        return node
    if not isinstance(node, TensorBox):
        node = TensorBox(node)
    if _sym(strides[0]) == 0 and _sym(sizes[0]) > 1:
        first = _pad_head_dim(lowerings[aten.slice](node, 0, 0, 1), width)
        return lowerings[aten.expand](first, [*sizes[:-1], width])
    padded = constant_pad_nd(node, [0, width - head_dim])
    return ExternKernel.require_exact_strides(
        padded, infer_dense_strides([*sizes[:-1], width], strides)
    )


def _reads_in_place(node, width: int) -> bool:
    """Whether the forward can read K or V at its stored width through a
    ``width``-column view (``kv_head_dims``)."""
    stride = _get_supported_bhsd_stride(node, broadcast_batch=True)
    if stride is None:
        return False
    sizes = _syms(node.get_size())
    if sizes[-1] % 8 or any(bool(value % 8 != 0) for value in stride[:-1]):
        return False
    # Missing columns load from up to row 2 * Sk + width / 8 of each head's
    # buffer view.
    rows = 2 * sizes[2] + width // 8 + 1
    element_size = torch._utils._element_size(node.get_dtype())
    return rows * stride[2] * element_size < _MAX_BUFFER_BYTES


def _unpad_head_dim(node, width: int, stride=None):
    """Return the first ``width`` head-dim columns of a padded result, copied to
    ``stride`` if given."""
    if V.graph.sizevars.guard_int(node.get_size()[-1]) == width:
        return node
    columns = lowerings[aten.slice](node, 3, 0, width)
    if stride is None:
        return columns
    return ExternKernel.require_exact_strides(columns, stride)


def _bhsd_sizes(node):
    """B, H, S, D of a 4D tensor: H and D specialized, B and S as SymInts."""
    batch, heads, length, dim = node.get_size()
    return (
        _sym(batch),
        V.graph.sizevars.guard_int(heads),
        _sym(length),
        V.graph.sizevars.guard_int(dim),
    )


def _broadcast_kv_batch(config, batch, kv_batch):
    """Read the single K/V batch for every query batch through a zero stride."""
    if kv_batch == 1 < batch:
        for name in ("K_STRIDE", "V_STRIDE"):
            config[name] = (0, *config[name][1:])


def _fits_u32_head_slice(node, heads=1) -> bool:
    try:
        sizes = _syms(node.get_size())
        strides = _syms(node.get_stride())
        element_size = torch._utils._element_size(node.get_dtype())
    except (AttributeError, TypeError, ValueError):
        return False
    if len(sizes) != 4 or len(strides) != 4 or any(stride < 0 for stride in strides):
        return False
    storage_elements = 1 + sum(
        (size - 1) * stride for size, stride in zip((heads, *sizes[-2:]), strides[-3:])
    )
    return storage_elements * element_size < _MAX_BUFFER_BYTES


def _is_contiguous_shape_stride(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> bool:
    if len(shape) != len(stride):
        return False
    expected = _contiguous_strides(tuple(max(size, 1) for size in shape))
    return all(
        actual == contiguous or size == 1
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
        shapes = tuple(_syms(node.get_size()) for node in nodes)
        full_numel = _sym(full_kv_num_blocks.get_numel())
        dtypes = tuple(node.get_dtype() for node in nodes)
        devices = tuple(node.get_device() for node in nodes)
    except (AttributeError, TypeError, ValueError):
        return None, f"{prefix}{static_reason}"

    if any(dtype != torch.int32 for dtype in dtypes):
        return None, f"{prefix}requires int32 BlockMask metadata"
    if any(device != query.get_device() for device in devices):
        return None, f"{prefix}requires BlockMask metadata on the query device"
    try:
        strides = tuple(_syms(node.get_stride()) for node in nodes)
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


def _sparse_block_size_reason(sparse_q_block_size, sparse_kv_block_size) -> str:
    sizes = (sparse_q_block_size, sparse_kv_block_size)
    if sizes == (128, 128):
        return ""
    if all(size > 0 for size in sizes):
        return "requires BlockMask metadata that splits or merges into 128x128 blocks"
    return f"requires positive sparse Q/KV block sizes, got {sizes}"


@dataclass(frozen=True)
class _BlockStates:
    """The int8 [B, H, Q blocks, KV blocks] BLOCK_STATE_* of a merged BlockMask,
    one per block of ``block_sizes``."""

    buffer: Any
    block_sizes: tuple[int, int]


def _mask_captures(mask_mod_other_buffers, block_states):
    """The kernels' mask captures: the mask_mod's, then any block states."""
    captures = mask_capture_tensors(mask_mod_other_buffers)
    if block_states is None:
        return captures
    return [*captures, block_states.buffer]


def _get_mask_program_config(
    mask_graph, mask_mod_other_buffers, block_states=None, query=None
):
    """``block_states`` makes the program drop the empty blocks of a merged
    BlockMask and keep its full ones; ``query`` gives the Q heads it reads."""
    trivial = is_trivial_mask_graph(mask_graph.graph_module)
    if trivial and block_states is None:
        return {
            "MASK_PROGRAM": (),
            "MASK_PROGRAM_OUTPUT": 0,
            "MASK_BUFFER_COUNT": 0,
            "MASK_BUFFER_SHAPES": (),
            "MASK_BUFFER_STRIDES": (),
            "MASK_BUFFER_DTYPES": (),
        }, ""
    mask_program = None
    if not trivial:
        mask_program, reason = lower_flydsl_mask_graph(
            mask_graph.graph_module, mask_mod_other_buffers
        )
        if mask_program is None:
            return None, f"unsupported mask_mod: {reason}"
    if block_states is not None:
        if query is None:
            return None, "requires backed BlockMask dimensions"
        try:
            num_q_heads = V.graph.sizevars.guard_int(query.get_size()[1])
            shape = list(_syms(block_states.buffer.get_size()))
            shape[1] = V.graph.sizevars.guard_int(block_states.buffer.get_size()[1])
            strides = _syms(block_states.buffer.get_stride())
        except (AttributeError, TypeError, ValueError):
            return None, "requires backed BlockMask dimensions"
        # Kernels address captures with i32 element offsets.
        if math.prod(shape) >= 1 << 31:
            return None, "requires fewer than 2**31 merged BlockMask block states"
        # Extents that follow Sq or Sk are read by the kernels at runtime.
        shape = tuple(map(_template_int, shape))
        strides = tuple(map(_template_int, strides))
        if num_q_heads % shape[1]:
            return None, "BlockMask head dimension must be 1, Hkv, or Hq"
        mask_program = add_block_states(
            mask_program,
            shape=shape,
            strides=strides,
            block_sizes=block_states.block_sizes,
            head_group=num_q_heads // shape[1],
        )
    if mask_program is None:
        raise AssertionError("expected a lowered mask program")
    return {
        "MASK_PROGRAM": mask_program.instructions,
        "MASK_PROGRAM_OUTPUT": mask_program.output,
        "MASK_BUFFER_COUNT": mask_program.buffer_count,
        "MASK_BUFFER_SHAPES": mask_program.buffer_shapes,
        "MASK_BUFFER_STRIDES": mask_program.buffer_strides,
        "MASK_BUFFER_DTYPES": mask_program.buffer_dtypes,
    }, ""


def _get_score_mod_config(subgraph, score_mod_other_buffers, query, mask_buffer_count):
    """Gate a score_mod's captures; unsupported ops are refused when rendering.
    ``CAPTURE_TABLE``: the captures exceed the kernels' direct capture slots."""
    if is_trivial_score_graph(subgraph.graph_module):
        return (
            {
                "HAS_SCORE_MOD": False,
                "CAPTURE_TABLE": mask_buffer_count > MAX_DIRECT_CAPTURES,
            },
            [],
            "",
        )
    tensors = [
        buffer
        for buffer in score_mod_other_buffers
        if not isinstance(buffer, sympy.Expr)
    ]
    for buffer in tensors:
        if not buffer.get_size():
            # Scalars are copied to a [1] fp32 device buffer, see
            # _prepare_score_mod_captures.
            if not buffer.get_dtype().is_floating_point:
                return (
                    None,
                    [],
                    f"requires floating-point scalar captures, got {buffer.get_dtype()}",
                )
            continue
        if buffer.get_dtype() not in _SCORE_MOD_CAPTURE_DTYPES:
            return (
                None,
                [],
                "requires float32/bfloat16/float16 score_mod captures, got "
                f"{buffer.get_dtype()}",
            )
        if buffer.get_device() != query.get_device():
            return None, [], "requires score_mod captures on the query device"
        try:
            sizes = [V.graph.sizevars.guard_int(s) for s in buffer.get_size()]
            strides = [V.graph.sizevars.guard_int(s) for s in buffer.get_stride()]
        except (AttributeError, TypeError, ValueError):
            return None, [], "requires static score_mod capture shapes and strides"
        if len(sizes) > 4 or any(stride < 0 for stride in strides):
            return None, [], "requires score_mod captures of rank <= 4"
        extent = 1 + sum((size - 1) * stride for size, stride in zip(sizes, strides))
        if extent * buffer.get_dtype().itemsize >= _MAX_BUFFER_BYTES:
            return None, [], "requires score_mod captures smaller than 4 GiB"
    return (
        {
            "HAS_SCORE_MOD": True,
            "SCORE_MOD_BUFFER_COUNT": len(tensors),
            # 0-d captures of other float dtypes are passed as float32.
            "SCORE_MOD_BUFFER_DTYPES": tuple(
                str(
                    buffer.get_dtype()
                    if buffer.get_dtype() in _SCORE_MOD_CAPTURE_DTYPES
                    else torch.float32
                ).removeprefix("torch.")
                for buffer in tensors
            ),
            "CAPTURE_TABLE": mask_buffer_count + len(tensors) > MAX_DIRECT_CAPTURES,
        },
        tensors,
        "",
    )


def _make_capture_table(count, device):
    """An int64 buffer the generated launcher fills with capture base addresses."""
    return empty_strided([count], [1], dtype=torch.int64, device=device)


def _capture_table_reason(mask_config, capture_table) -> str:
    """Why a capture table cannot take these mask captures: each one with a
    runtime extent also takes a capture slot after the table."""
    runtime_extents = sum(
        _is_runtime(shape) or _is_runtime(strides)
        for shape, strides in zip(
            mask_config["MASK_BUFFER_SHAPES"], mask_config["MASK_BUFFER_STRIDES"]
        )
    )
    if capture_table and runtime_extents >= MAX_DIRECT_CAPTURES:
        return (
            f"requires at most {MAX_DIRECT_CAPTURES - 1} captures with runtime "
            "extents next to a capture table"
        )
    return ""


def _prepare_captures(
    score_mod_other_buffers, mask_mod_other_buffers, block_states, device
):
    score_mod_other_buffers = maybe_realize(list(score_mod_other_buffers or []))
    mask_mod_other_buffers = maybe_realize(
        prepare_flydsl_mask_captures(list(mask_mod_other_buffers or []), device)
    )
    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)
    if block_states is not None:
        (state_buffer,) = maybe_realize([block_states.buffer])
        block_states = _BlockStates(state_buffer, block_states.block_sizes)
    return score_mod_other_buffers, mask_mod_other_buffers, block_states


def _prepare_score_mod_captures(tensors, device):
    """Pass 0-d captures as [1] fp32 device tensors; FlyDSL kernel arguments need a dim."""
    from ...lowering import to_device, to_dtype, view

    prepared = []
    for buffer in tensors:
        if not buffer.get_size():
            if buffer.get_device() != device:
                buffer = to_device(buffer, device)
            if buffer.get_dtype() not in _SCORE_MOD_CAPTURE_DTYPES:
                buffer = to_dtype(buffer, torch.float32)
            (buffer,) = maybe_realize([view(buffer, [1])])
        prepared.append(buffer)
    return prepared


def _is_gfx950_device(device) -> bool:
    if not flydsl_utils.runtime_available() or not torch.cuda.is_available():
        return False
    from ...codegen.flydsl.flydsl_scheduling import _get_flydsl_device_arch

    index = device.index if device.index is not None else torch.cuda.current_device()
    return (
        _get_flydsl_device_arch(index) == FLEX_ARCH
        and torch.cuda.get_device_properties(index).warp_size == WAVE_SIZE
    )


def _target_limits(node) -> TargetLimits:
    """The ``TargetLimits`` of ``node``'s device."""
    return target_limits(node.get_device())


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
    allow_score_mod: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    device = query.get_device()
    if device is None or device.type != "cuda" or not _is_gfx950_device(device):
        return None, "requires ROCm gfx950 and the FlyDSL runtime"
    if query.get_dtype() not in _FLYDSL_DTYPES:
        return None, f"supports BF16 and FP16 only, got {query.get_dtype()}"
    if query.get_dtype() != key.get_dtype() or query.get_dtype() != value.get_dtype():
        return None, "requires query, key, and value to have the same dtype"
    if not allow_score_mod:
        if not is_trivial_score_graph(subgraph.graph_module):
            return None, "supports identity score_mod only"
        if score_mod_other_buffers:
            return None, "does not support captured score_mod buffers"
    if mask_capture_tensors(mask_mod_other_buffers) and not allow_mask_mod_buffers:
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
        else _get_supported_bhsd_stride(
            node, broadcast_batch=name in ("K_STRIDE", "V_STRIDE")
        )
        for name, node in named_tensors
    )
    if any(
        node is not None and stride is None
        for (_, node), stride in zip(named_tensors, strides, strict=True)
    ):
        return None, "requires 4D BHSD tensors with contiguous head dimensions"
    if not all(
        _fits_u32_head_slice(node) for _, node in named_tensors if node is not None
    ):
        return None, "requires every per-head tensor slice to be smaller than 4 GiB"

    return {
        **{
            name: stride
            for (name, _), stride in zip(named_tensors, strides, strict=True)
        },
        "DTYPE": _FLYDSL_DTYPES[query.get_dtype()],
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
    kv_head_dims=None,
    qo_in_place=False,
    block_states=None,
    head_slow_order=False,
    kv_splits=1,
    single_pass=False,
) -> tuple[dict[str, Any] | None, str]:
    """``kv_head_dims``: the stored K/V widths, read in place at ``query``'s
    padded head dims, or with ``qo_in_place`` at the padded dims that cover
    them, Q and O then also kept at their stored widths.
    ``block_states``: see ``_maybe_split_block_mask``.
    ``head_slow_order``: allow head-slow workgroup order, for metadata that
    lists every block as full. ``kv_splits``: see ``_forward_kv_splits``.
    ``single_pass`` runs V in one launch where ``forward_single_pass`` holds
    and no capture table is read."""
    if full_kv_num_blocks is None or full_kv_indices is None:
        return None, "requires full_kv_num_blocks/full_kv_indices metadata"

    try:
        (b, hq, sq, qk_dim), (bkv, hkv, sk, key_dim), (bv, hv, sv, v_dim) = (
            _bhsd_sizes(node) for node in (query, key, value)
        )
        if qo_in_place and (kv_head_dims is None or qk_dim != kv_head_dims[0]):
            return None, "keeps Q and O at their stored widths only with kv_head_dims"
        if kv_head_dims is not None:
            if (key_dim, v_dim) != tuple(kv_head_dims):
                return None, "received K/V widths that differ from kv_head_dims"
            padded = kernel_head_dims(key_dim, v_dim)
            if padded is None:
                return None, "kv_head_dims must require head-dimension padding"
            key_dim, v_dim = padded
        # The Q/O widths in memory; split-KV partials keep the padded V width.
        stored_dims = (qk_dim, v_dim)
        if qo_in_place:
            stored_dims = (
                qk_dim,
                kv_head_dims[1]
                if kv_head_dims is not None and kv_splits == 1
                else v_dim,
            )
            qk_dim = key_dim
        sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)
        sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
        scale_value = float(scale)
        output_stride = _syms(
            infer_dense_strides(
                [query.get_size()[0], hq, query.get_size()[2], stored_dims[1]],
                query.get_stride(),
            )
        )
    except (AttributeError, TypeError, ValueError):
        return None, "requires backed tensor and BlockMask dimensions"

    metadata, reason = _get_block_mask_metadata(
        query,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        static_reason="requires backed tensor and BlockMask dimensions",
    )
    if metadata is None:
        return None, reason
    mask_shape = metadata.count_shape
    index_shape = metadata.index_shape
    full_count_shape = metadata.full_count_shape
    full_index_shape = metadata.full_index_shape

    mask_config, reason = _get_mask_program_config(
        mask_graph, mask_mod_other_buffers, block_states, query
    )
    if mask_config is None:
        return None, reason

    common_config, reason = _get_flydsl_common_config(
        query=query,
        key=key,
        value=value,
        subgraph=subgraph,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
        allow_mask_mod_buffers=bool(mask_config["MASK_BUFFER_COUNT"]),
        allow_score_mod=True,
    )
    if common_config is None:
        return None, reason
    score_mod_config, _, reason = _get_score_mod_config(
        subgraph, score_mod_other_buffers, query, mask_config["MASK_BUFFER_COUNT"]
    )
    if score_mod_config is None:
        return None, f"unsupported score_mod: {reason}"

    if (bkv, hkv, sk) != (bv, hv, sv):
        return None, "requires key and value to have matching B/Hkv/Sk dimensions"
    if bkv not in (1, b):
        return None, "requires a K/V batch of 1 or B"
    _broadcast_kv_batch(common_config, b, bkv)
    if qk_dim != key_dim:
        return None, "requires query and key to have the same head dimension"
    reason = _head_dim_reason(qk_dim, v_dim)
    if reason:
        return None, reason
    target = _target_limits(query)
    value_slice = forward_value_slice(qk_dim, v_dim, target)
    single_pass = (
        single_pass
        and not score_mod_config["CAPTURE_TABLE"]
        and forward_single_pass(qk_dim, v_dim, target)
    )
    if single_pass:
        value_slice = v_dim
    if value_slice != v_dim and kv_head_dims is not None:
        return None, "reads K/V in place only with V in one kernel launch"
    if hkv <= 0 or hq % hkv != 0 or sq <= 0 or sk <= 0:
        return None, "requires positive lengths and Hq divisible by Hkv"
    decode = 0 < sq < 128
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
    supports_prefill = not decode and mask_shape[2] == (sq + 127) // 128
    supports_decode = (
        decode
        and mask_shape[1] in (1, hkv)
        and mask_shape[2] == 1
        and 0 < packed_decode_rows <= 256
    )
    reason = _sparse_block_size_reason(sparse_q_block_size, sparse_kv_block_size)
    if reason:
        return None, reason
    if not has_full_blocks or max_full_blocks <= 0 or index_shape[-1] <= 0:
        return None, "requires non-empty partial and full BlockMask storage"
    if not (supports_prefill or supports_decode):
        return (
            None,
            "requires prefill Sq >= 128 with ceil(Sq / 128) BlockMask rows, "
            "or decode 0 < Sq < 128 with a shared/per-KV-head BlockMask and "
            "(Hq/Hkv)*Sq <= 256 (the lowering merges per-Q-head decode masks "
            "per KV head)",
        )
    # Decode reads the query rows of a whole GQA group through one buffer view.
    if decode and not _fits_u32_head_slice(query, gqa_group_size):
        return None, "requires every GQA group's query slice to be smaller than 4 GiB"
    if kv_splits > 1:
        output_stride = _contiguous_strides((b * kv_splits, hq, sq, v_dim))
    # With V wider than QK the output's slices can outgrow the query's.
    output_elements = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(
            (gqa_group_size if decode else 1, sq, stored_dims[1]), output_stride[1:]
        )
    )
    if output_elements * torch._utils._element_size(query.get_dtype()) >= (
        _MAX_BUFFER_BYTES
    ):
        return None, "requires every per-head output slice to be smaller than 4 GiB"
    try:
        plan = forward_plan(
            batch_size=b,
            num_q_heads=hq,
            num_kv_heads=hkv,
            seq_q=sq,
            seq_kv=sk,
            qk_head_dim=qk_dim,
            v_head_dim=value_slice,
            has_score_mod=score_mod_config["HAS_SCORE_MOD"],
            padded_kv=kv_head_dims is not None,
            has_mask_buffers=bool(mask_config["MASK_BUFFER_COUNT"]),
            kv_index_widths=(index_shape[-1], max_full_blocks),
            **({"single_pass": True} if single_pass else {}),
            target=target,
        )
    except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
        return None, "requires backed tensor dimensions"
    sizes = {
        "BATCH_SIZE": _template_int(b),
        "SEQ_Q": _template_int(sq),
        "SEQ_KV": _template_int(sk),
        "BLOCK_MASK_BATCH": _template_int(mask_shape[0]),
        "NUM_Q_BLOCKS": _template_int(mask_shape[2]),
        "MAX_PARTIAL_BLOCKS": _template_int(index_shape[-1]),
        "MAX_FULL_BLOCKS": _template_int(max_full_blocks),
        **{
            name: _template_strides(common_config.pop(name), length)
            for name, length in (("Q_STRIDE", sq), ("K_STRIDE", sk), ("V_STRIDE", sk))
        },
        "O_STRIDE": _template_strides(output_stride, sq),
    }
    runtime = _is_runtime(list(sizes.values())) or _is_runtime(
        mask_config["MASK_BUFFER_SHAPES"]
    )
    reason = _capture_table_reason(mask_config, score_mod_config["CAPTURE_TABLE"])
    if reason:
        return None, reason

    return (
        {
            **sizes,
            "NUM_Q_HEADS": hq,
            "NUM_KV_HEADS": hkv,
            "QK_HEAD_DIM": qk_dim,
            "V_HEAD_DIM": value_slice,
            **({} if value_slice == v_dim else {"VALUE_SLICES": v_dim // value_slice}),
            **({"SINGLE_PASS": True} if single_pass else {}),
            "BLOCK_MASK_HEADS": mask_shape[1],
            "SPARSE_Q_BLOCK_SIZE": sparse_q_block_size,
            "SPARSE_KV_BLOCK_SIZE": sparse_kv_block_size,
            **mask_config,
            **score_mod_config,
            **common_config,
            "SM_SCALE": scale_value,
            "RUNTIME_SIZES": runtime,
            "TARGET_LIMITS": tuple(target),
            **({"PLAN": plan} if runtime else {}),
            **({} if kv_head_dims is None else {"KV_HEAD_DIMS": tuple(kv_head_dims)}),
            **({"QO_HEAD_DIMS": stored_dims} if qo_in_place else {}),
            # The block-state lookup schedules a packed probability convert
            # within two wait states of the PV MFMA that reads it.
            **({} if block_states is None else {"TIE_STREAMED_PROBABILITIES": True}),
            **(
                {"HEAD_SLOW_ORDER": True}
                if head_slow_order and use_head_slow_order(hq, sq)
                else {}
            ),
            **({} if kv_splits == 1 else {"KV_SPLITS": kv_splits}),
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
    subgraph_buffer=None,
    score_mod_inputs=None,
    capture_table=None,
    kv_head_dims=None,
    qo_in_place=False,
    block_states=None,
    head_slow_order=False,
    kv_splits=1,
    single_pass=False,
) -> tuple[bool, str]:
    """Offer the general forward after validating layouts, masks and captures."""
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
        kv_head_dims=kv_head_dims,
        qo_in_place=qo_in_place,
        block_states=block_states,
        head_slow_order=head_slow_order,
        kv_splits=kv_splits,
        single_pass=single_pass,
    )
    if config is None:
        return False, reason

    config["WRITE_MAX_SCORES"] = bool(write_max_scores)
    mod_inputs, mod_kwargs, reason = _mod_inputs(
        config,
        query,
        subgraph,
        subgraph_buffer,
        score_mod_other_buffers,
        mask_mod_other_buffers,
        block_states,
        score_mod_inputs,
    )
    if mod_inputs is None:
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
        *mod_inputs,
    ]
    mutated_inputs = [logsumexp, max_scores]
    if config["CAPTURE_TABLE"]:
        if capture_table is None:
            capture_table = _make_capture_table(
                config["MASK_BUFFER_COUNT"] + config.get("SCORE_MOD_BUFFER_COUNT", 0),
                query.get_device(),
            )
        input_nodes.append(capture_table)
        mutated_inputs.append(capture_table)
    return _append_template_choice(
        flex_flydsl_forward_template,
        choices,
        input_nodes=input_nodes,
        mutated_inputs=mutated_inputs,
        layout=layout,
        **mod_kwargs,
        **config,
    )


def _mod_inputs(
    config,
    query,
    subgraph,
    subgraph_buffer,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    block_states,
    score_mod_inputs,
):
    """(capture inputs, template kwargs, reason) of a config's mask_mod and
    score_mod; the inputs are None if they cannot be lowered."""
    inputs: list[Any] = []
    mod_kwargs: dict[str, Any] = {}
    mask_buffer_count = config["MASK_BUFFER_COUNT"]
    if mask_buffer_count:
        mask_captures = _mask_captures(mask_mod_other_buffers, block_states)
        if len(mask_captures) != mask_buffer_count:
            return None, mod_kwargs, "mask_mod capture count changed during lowering"
        inputs.extend(mask_captures)
    if config["HAS_SCORE_MOD"]:
        if subgraph_buffer is None:
            return None, mod_kwargs, "score_mod requires its lowered subgraph"
        _, score_mod_tensors, _ = _get_score_mod_config(
            subgraph, score_mod_other_buffers, query, mask_buffer_count
        )
        if score_mod_inputs is None:
            score_mod_inputs = _prepare_score_mod_captures(
                score_mod_tensors, query.get_device()
            )
        inputs.extend(score_mod_inputs)
        # The rendered mods match loads to aux_buffers by name.
        mod_kwargs = {"subgraphs": [subgraph_buffer], "aux_buffers": score_mod_tensors}
    return inputs, mod_kwargs, ""


def _append_template_choice(template, choices, **kwargs) -> tuple[bool, str]:
    choices_before = len(choices)
    error = template.maybe_append_choice(choices, **kwargs)
    if len(choices) == choices_before:
        return False, f"FlyDSL template registration failed: {error}"
    return True, ""


def _create_dense_metadata(query, key):
    """Materialize only the forward metadata for the frontend's no-mask sentinel."""
    num_q_blocks, num_kv_blocks = (
        _ceil_div(_size_expr(node.get_size()[2]), 128) for node in (query, key)
    )
    shape = [1, 1, num_q_blocks]
    device = query.get_device()
    if isinstance(num_q_blocks, int) and isinstance(num_kv_blocks, int):
        # Graph constants: four tiny kernels per call cost as much as a small
        # attention.
        with torch.utils._python_dispatch._disable_current_modes():
            tensors = (
                torch.zeros(shape, dtype=torch.int32, device=device),
                torch.zeros([*shape, 1], dtype=torch.int32, device=device),
                torch.full(shape, num_kv_blocks, dtype=torch.int32, device=device),
                torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
                .expand(*shape, num_kv_blocks)
                .contiguous(),
            )
        return tuple(V.graph.add_tensor_constant(tensor) for tensor in tensors)
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


# Merged-mask passes carry the full list in the low 16 bits, partial in the high.
_UNION_SHIFT = 16
_UNION_LOW = (1 << _UNION_SHIFT) - 1
MAX_UNION_KV_BLOCKS = (1 << 15) - 1


def _reduce_int32(
    device, inner_fn, ranges, reduction_ranges, reduction_type: ReductionType = "sum"
):
    result = Reduction.create(
        device=device,
        dst_dtype=torch.int32,
        src_dtype=torch.int32,
        inner_fn=inner_fn,
        ranges=ranges,
        reduction_ranges=reduction_ranges,
        reduction_type=reduction_type,
    )
    result.realize()
    return result


def _pack(full, partial, wide=False):
    """Pack two counts into an int32, or into an int64 if ``wide``."""
    if wide:
        full, partial = (ops.to_dtype(value, torch.int64) for value in (full, partial))
        return ops.add(
            full, ops.bitwise_left_shift(partial, ops.constant(32, torch.int64))
        )
    return ops.add(
        full, ops.bitwise_left_shift(partial, ops.constant(_UNION_SHIFT, torch.int32))
    )


def _unpack(packed, wide=False):
    """The int32 counts of ``_pack(..., wide)``."""
    if wide:
        return (
            _int32(ops.bitwise_and(packed, ops.constant((1 << 32) - 1, torch.int64))),
            _int32(ops.bitwise_right_shift(packed, ops.constant(32, torch.int64))),
        )
    return (
        ops.bitwise_and(packed, ops.constant(_UNION_LOW, torch.int32)),
        ops.bitwise_right_shift(packed, ops.constant(_UNION_SHIFT, torch.int32)),
    )


def _int32(value):
    return ops.to_dtype(value, torch.int32)


def _int32_value(value, dtype=torch.int32):
    """An int or size expression as a value of ``dtype``."""
    if isinstance(value, int):
        return ops.constant(value, dtype)
    return ops.index_expr(value, dtype)


def _create_packed_decode_union_metadata(
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    *,
    num_kv_heads,
    num_kv_blocks,
):
    """Merge a per-Q-head decode BlockMask over each GQA group: a block stays
    full only if every head lists it as full. Assumes distinct list entries."""
    mask_batch, num_q_heads, num_q_blocks = map(_size_expr, kv_num_blocks.get_size())
    num_q_heads = V.graph.sizevars.guard_int(num_q_heads)
    capacities = [
        _size_expr(node.get_size()[-1]) for node in (full_kv_indices, kv_indices)
    ]
    group = num_q_heads // num_kv_heads
    device = kv_num_blocks.get_device()
    lists = [
        (
            full_kv_num_blocks.make_loader(),
            full_kv_indices.make_loader(),
            capacities[0],
        ),
        (kv_num_blocks.make_loader(), kv_indices.make_loader(), capacities[1]),
    ]
    rows = [mask_batch, num_kv_heads, num_q_blocks]
    ranges = [*rows, num_kv_blocks]

    def hits_fn(index, reduction_index):
        batch, kv_head, q_block, kv_block = index
        member, slot = reduction_index
        row = [batch, kv_head * group + member, q_block]
        slot_value = ops.index_expr(slot, torch.int32)
        hits = []
        for load_count, load_index, capacity in lists:
            in_range = ops.lt(slot_value, _int32_value(capacity))
            listed = ops.masked(
                in_range,
                lambda: ops.eq(
                    load_index([*row, slot]), ops.index_expr(kv_block, torch.int32)
                ),
                False,
            )
            hits.append(_int32(ops.and_(ops.lt(slot_value, load_count(row)), listed)))
        return _pack(*hits)

    # Per KV block: how many group heads list it as full / as partial.
    load_hits = _reduce_int32(
        device, hits_fn, ranges, [group, _max_size(*capacities)]
    ).make_loader()

    def flags(index):
        full_hits, partial_hits = _unpack(load_hits(index))
        full = ops.eq(full_hits, ops.constant(group, torch.int32))
        listed = ops.ne(ops.add(full_hits, partial_hits), ops.constant(0, torch.int32))
        return _int32(full), _int32(ops.and_(listed, ops.logical_not(full)))

    def prefix_fn(index, reduction_index):
        *row, kv_block = index
        (other,) = reduction_index
        return ops.where(
            ops.le(
                ops.index_expr(other, torch.int32),
                ops.index_expr(kv_block, torch.int32),
            ),
            _pack(*flags([*row, other])),
            ops.constant(0, torch.int32),
        )

    # Inclusive per-list prefix counts of the merged flags.
    load_prefix = _reduce_int32(
        device, prefix_fn, ranges, [num_kv_blocks]
    ).make_loader()

    # The j-th listed block is the number of blocks whose prefix count is <= j.
    def position_fn(index, reduction_index):
        *row, slot = index
        (kv_block,) = reduction_index
        slot_value = ops.index_expr(slot, torch.int32)
        return _pack(
            *(
                _int32(ops.le(prefix, slot_value))
                for prefix in _unpack(load_prefix([*row, kv_block]))
            )
        )

    load_positions = _reduce_int32(
        device, position_fn, ranges, [num_kv_blocks]
    ).make_loader()
    outputs = []
    for part in range(2):
        counts = Pointwise.create(
            device=device,
            dtype=torch.int32,
            inner_fn=lambda index, part=part: _unpack(
                load_prefix([*index, num_kv_blocks - 1])
            )[part],
            ranges=rows,
        )
        # Slots past the count clamp to a valid block index.
        indices = Pointwise.create(
            device=device,
            dtype=torch.int32,
            inner_fn=lambda index, part=part: ops.minimum(
                _unpack(load_positions(index))[part], _int32_value(num_kv_blocks - 1)
            ),
            ranges=ranges,
        )
        counts.realize()
        indices.realize()
        outputs.append((counts, indices))
    (full_counts, full_indices), (partial_counts, partial_indices) = outputs
    return partial_counts, partial_indices, full_counts, full_indices


def _maybe_split_block_mask(
    query,
    key,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    sparse_q_block_size,
    sparse_kv_block_size,
):
    """Rewrite a BlockMask of any block sizes as 128x128 (splitting multiples of
    128, merging others); return the metadata, block sizes and ``_BlockStates``
    (None unless merged) the kernels see."""
    metadata = (kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices)
    unchanged = (*metadata, sparse_q_block_size, sparse_kv_block_size, None)
    try:
        block_sizes = [
            V.graph.sizevars.guard_int(size)
            for size in (sparse_q_block_size, sparse_kv_block_size)
        ]
        seq_q, seq_kv = (_sym(node.get_size()[2]) for node in (query, key))
        shapes = [list(_syms(node.get_size())) for node in metadata]
    except (AttributeError, TypeError, ValueError):
        return unchanged
    if block_sizes == [128, 128] or min(block_sizes) <= 0:
        return unchanged
    if any(node.get_dtype() != torch.int32 for node in metadata):
        return unchanged
    count_shape = shapes[0]
    if (
        len(count_shape) != 3
        or count_shape[2] != -(-seq_q // block_sizes[0])
        or shapes[2] != count_shape
        or any(len(shape) != 4 or shape[:3] != count_shape for shape in shapes[1::2])
    ):
        return unchanged
    merge = any(size % 128 for size in block_sizes)
    # Merged prefix counts pack both lists into an int32 up to this width.
    wide_counts = merge and bool(-(-seq_kv // 128) > MAX_UNION_KV_BLOCKS)
    seq_kv = _size_expr(key.get_size()[2])
    num_q_blocks, num_kv_blocks = (
        _ceil_div(_size_expr(length), 128) for length in (query.get_size()[2], seq_kv)
    )
    if merge:
        *merged, states = _merge_block_mask_metadata(
            *metadata,
            block_sizes=tuple(block_sizes),
            num_q_blocks=num_q_blocks,
            num_kv_blocks=num_kv_blocks,
            seq_kv=seq_kv,
            wide_counts=wide_counts,
        )
        state_sizes = tuple(map(_state_block_size, block_sizes))
        return (*merged, 128, 128, _BlockStates(states, state_sizes))
    expanded = _expand_block_mask_metadata(
        *metadata,
        q_factor=block_sizes[0] // 128,
        kv_factor=block_sizes[1] // 128,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
    )
    return (*expanded, 128, 128, None)


def _scatter(buffer, ranges, value_fn, offset_fn):
    """Store ``value_fn(index)`` at flat ``offset_fn(index)`` of a realized 1D
    ``buffer`` for every index of ``ranges``."""
    from ... import ir

    size = _size_expr(buffer.get_size()[0])
    scatter = ir.Scatter(
        device=buffer.get_device(),
        dtype=buffer.get_dtype(),
        inner_fn=value_fn,
        ranges=ranges,
        output_indexer=lambda index: [
            ops.indirect_indexing(offset_fn(index), size, wrap_neg=False)
        ],
        scatter_mode=None,
    )
    mutation = ir.ComputedBuffer(
        name=None, layout=ir.MutationLayoutSHOULDREMOVE(buffer), data=scatter
    )
    mutation.name = V.graph.register_buffer(mutation)
    V.graph.register_operation(mutation)


def _overlapped_blocks(size):
    """The most blocks of ``size`` that one 128-wide block overlaps."""
    if size % 128 == 0 or 128 % size == 0:
        return max(1, 128 // size)
    # 128-wide blocks start at multiples of gcd(size, 128) within a block.
    return (size - math.gcd(size, 128) + 127) // size + 1


def _state_block_size(size):
    """The block size of the merged states of BlockMask blocks of ``size``: for
    a size that does not tile 128, its largest power-of-two factor if >= 16, so
    kernels find states with a shift."""
    step = size & -size
    if size % 128 and 128 % size and step >= 16:
        return step
    return size


def _flat_offset(row, shape, last):
    """The int64 row-major offset of ``(*row, last)`` in ``shape``."""
    offset = sympy.Integer(0)
    for position, size in zip(row, shape[1:]):
        offset = (offset + position) * size
    return ops.add(ops.index_expr(offset, torch.int64), ops.to_dtype(last, torch.int64))


def _scatter_block_states(
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    *,
    state_shape,
    num_user_kv_blocks,
    repeats=(1, 1),
):
    """The int8 ``state_shape`` BLOCK_STATE_* of BlockMask lists, ``repeats``
    states per BlockMask block in each of the Q and KV dims."""
    from ...lowering import view

    mask_batch, mask_heads, user_q_blocks = map(_size_expr, kv_num_blocks.get_size())
    device = kv_num_blocks.get_device()
    repeats = list(repeats)
    state_numel = math.prod(state_shape)
    # One slot past the states takes the stores of unlisted slots.
    flat_states = full([state_numel + 1], 0, dtype=torch.int8, device=device)
    flat_states.realize()

    constant = _int32_value

    # The full list is scattered last, so it wins for a block in both lists.
    for state, (counts, indices) in (
        (BLOCK_STATE_PARTIAL, (kv_num_blocks, kv_indices)),
        (BLOCK_STATE_FULL, (full_kv_num_blocks, full_kv_indices)),
    ):

        def state_offset(
            index, load_count=counts.make_loader(), load_index=indices.make_loader()
        ):
            *row, slot = index[:4]
            block = load_index(index[:4])
            listed = ops.and_(
                ops.lt(ops.index_expr(slot, torch.int32), load_count(row)),
                ops.and_(
                    ops.ge(block, constant(0)),
                    ops.lt(block, constant(num_user_kv_blocks)),
                ),
            )
            if repeats == [1, 1]:
                return ops.where(
                    listed,
                    _flat_offset(row, state_shape, block),
                    constant(state_numel, torch.int64),
                )
            # States of the block past the 128x128 grid are dropped.
            q_part, kv_part = index[4:]
            row = [*row[:2], row[2] * repeats[0] + q_part]
            kv_state = ops.add(
                ops.mul(block, constant(repeats[1])),
                ops.index_expr(kv_part, torch.int32),
            )
            listed = ops.and_(
                listed,
                ops.and_(
                    ops.lt(
                        ops.index_expr(row[2], torch.int32), constant(state_shape[2])
                    ),
                    ops.lt(kv_state, constant(state_shape[3])),
                ),
            )
            return ops.where(
                listed,
                _flat_offset(row, state_shape, kv_state),
                constant(state_numel, torch.int64),
            )

        slots = [
            mask_batch,
            mask_heads,
            user_q_blocks,
            _size_expr(indices.get_size()[-1]),
        ]
        _scatter(
            flat_states,
            slots if repeats == [1, 1] else [*slots, *repeats],
            lambda index, state=state: constant(state, torch.int8),
            state_offset,
        )
    states = view(lowerings[aten.slice](flat_states, 0, 0, state_numel), state_shape)
    return states


def _merge_block_mask_metadata(
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    *,
    block_sizes,
    num_q_blocks,
    num_kv_blocks,
    seq_kv,
    wide_counts=False,
):
    """Merge BlockMask lists of ``block_sizes`` into ascending 128x128 lists.

    A 128x128 block is full if every block it overlaps is, empty if none is
    listed, else partial. Also returns the int8 BLOCK_STATE_* of every block of
    ``_state_block_size`` over the 128x128 grid. ``wide_counts`` packs prefix
    counts into int64, needed past ``MAX_UNION_KV_BLOCKS`` 128-column blocks.
    """
    from ...lowering import view

    mask_batch, mask_heads, user_q_blocks = map(_size_expr, kv_num_blocks.get_size())
    user_blocks = (user_q_blocks, _ceil_div(seq_kv, block_sizes[1]))
    spans = [_overlapped_blocks(size) for size in block_sizes]
    # Every 128-wide block overlaps whole blocks at fixed offsets.
    aligned = all(size % 128 == 0 or 128 % size == 0 for size in block_sizes)
    device = kv_num_blocks.get_device()
    rows = [mask_batch, mask_heads, num_q_blocks]
    ranges = [*rows, num_kv_blocks]
    state_sizes = [_state_block_size(size) for size in block_sizes]
    # States per BlockMask block in each dim.
    repeats = [size // step for size, step in zip(block_sizes, state_sizes)]
    state_shape = [
        mask_batch,
        mask_heads,
        *(
            _ceil_div(blocks * 128, size)
            for blocks, size in zip((num_q_blocks, num_kv_blocks), state_sizes)
        ),
    ]
    states = _scatter_block_states(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        state_shape=state_shape,
        num_user_kv_blocks=user_blocks[1],
        repeats=repeats,
    )
    constant = _int32_value
    load_state = states.make_loader()

    def merge_fn(index, reduction_index):
        *outer, q_block, kv_block = index
        covered, overlaps = [], []
        for block, offset, size, span in zip(
            (q_block, kv_block), reduction_index, block_sizes, spans
        ):
            if size % 128 == 0:
                covered.append(FloorDiv(block, size // 128))
            elif 128 % size == 0:
                covered.append(block * span + offset)
            else:
                covered.append(FloorDiv(block * 128, size) + offset)
                last = FloorDiv(block * 128 + 127, size)
                overlaps.append(
                    ops.le(
                        ops.index_expr(covered[-1], torch.int32),
                        ops.index_expr(last, torch.int32),
                    )
                )
        inside = ops.and_(
            *(
                ops.lt(ops.index_expr(block, torch.int32), constant(count))
                for block, count in zip(covered, user_blocks)
            )
        )
        if aligned:
            block_state = ops.to_dtype(load_state([*outer, *covered]), torch.int32)
            full = ops.or_(
                ops.logical_not(inside), ops.eq(block_state, constant(BLOCK_STATE_FULL))
            )
            listed = ops.and_(inside, ops.ne(block_state, constant(BLOCK_STATE_EMPTY)))
            return _pack(_int32(full), _int32(listed))
        overlapped = functools.reduce(ops.and_, overlaps, inside)
        first_states = [block * repeat for block, repeat in zip(covered, repeats)]
        block_state = ops.masked(
            overlapped,
            lambda: ops.to_dtype(load_state([*outer, *first_states]), torch.int32),
            BLOCK_STATE_EMPTY,
        )
        not_full = ops.and_(overlapped, ops.ne(block_state, constant(BLOCK_STATE_FULL)))
        listed = ops.and_(overlapped, ops.ne(block_state, constant(BLOCK_STATE_EMPTY)))
        return _pack(_int32(not_full), _int32(listed))

    # Per 128x128 block: how many overlapped blocks are full (or, unaligned,
    # not full) and how many are listed.
    load_merged = _reduce_int32(device, merge_fn, ranges, spans).make_loader()

    def flags(index):
        hits, listed = _unpack(load_merged(index))
        full = ops.eq(hits, constant(spans[0] * spans[1] if aligned else 0))
        partial = ops.and_(ops.ne(listed, constant(0)), ops.logical_not(full))
        return _int32(full), _int32(partial)

    count_dtype = torch.int64 if wide_counts else torch.int32
    prefix = lowerings[aten.cumsum](
        Pointwise.create(
            device=device,
            dtype=count_dtype,
            inner_fn=lambda index: _pack(
                *cast(tuple[Any, Any], flags(index)), wide_counts
            ),
            ranges=ranges,
        ),
        3,
        dtype=count_dtype,
    )
    prefix.realize()
    load_prefix = prefix.make_loader()
    outputs = []
    for part in range(2):

        def count_fn(index, reduction_index, part=part):
            return flags([*index, *reduction_index])[part]

        counts = _reduce_int32(device, count_fn, rows, [num_kv_blocks])
        flat_indices = empty_strided(
            [math.prod(ranges)], [1], dtype=torch.int32, device=device
        )
        load_count = counts.make_loader()

        # Listed blocks come first, then the other blocks, so every slot past
        # the count still holds a valid block index.
        def list_offset(index, part=part, load_count=load_count):
            *row, kv_block = index
            listed_before = _unpack(load_prefix(index), wide_counts)[part]
            slot = ops.where(
                ops.ne(flags(index)[part], constant(0)),
                ops.sub(listed_before, constant(1)),
                ops.sub(
                    ops.add(load_count(row), ops.index_expr(kv_block, torch.int32)),
                    listed_before,
                ),
            )
            return _flat_offset(row, ranges, slot)

        _scatter(
            flat_indices,
            ranges,
            lambda index: ops.index_expr(index[-1], torch.int32),
            list_offset,
        )
        outputs.append((counts, view(flat_indices, ranges)))
    (full_counts, full_indices), (partial_counts, partial_indices) = outputs
    return partial_counts, partial_indices, full_counts, full_indices, states


def _expand_block_mask_metadata(
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    *,
    q_factor,
    kv_factor,
    num_q_blocks,
    num_kv_blocks,
):
    """Rewrite BlockMask lists of (128 * q_factor, 128 * kv_factor) blocks as
    128x128 lists (block j becomes tiles j * kv_factor + t), dropping the last
    block's tiles at or past ``num_kv_blocks``."""
    mask_batch, mask_heads, _ = map(_size_expr, kv_num_blocks.get_size())
    device = kv_num_blocks.get_device()
    rows = [mask_batch, mask_heads, num_q_blocks]
    last_block = (
        (num_kv_blocks - 1) // kv_factor
        if isinstance(num_kv_blocks, int)
        else _size_expr(FloorDiv(num_kv_blocks - 1, kv_factor))
    )
    # Tiles of the last block that lie before ceil(Sk / 128).
    tail = _size_expr(num_kv_blocks - last_block * kv_factor)
    # A runtime Sk may leave the last block partial.
    ragged_tail = not isinstance(tail, int) or tail < kv_factor

    def user_row(index):
        batch, head, q_block = index
        return [batch, head, FloorDiv(q_block, q_factor)]

    constant = _int32_value

    def expand(counts, indices):
        capacity = _size_expr(indices.get_size()[-1])
        load_count = counts.make_loader()
        load_index = indices.make_loader()
        load_last_slot = None
        if ragged_tail:

            def last_slot_fn(index, reduction_index):
                (slot,) = reduction_index
                row = user_row(index)
                slot_value = ops.index_expr(slot, torch.int32)
                listed = ops.and_(
                    ops.lt(slot_value, load_count(row)),
                    ops.eq(load_index([*row, slot]), constant(last_block)),
                )
                return ops.where(listed, slot_value, constant(capacity))

            load_last_slot = _reduce_int32(
                device, last_slot_fn, rows, [capacity], "min"
            ).make_loader()

        def count_fn(index):
            count = ops.mul(load_count(user_row(index)), constant(kv_factor))
            if load_last_slot is None:
                return count
            dropped = ops.where(
                ops.lt(load_last_slot(index), constant(capacity)),
                constant(kv_factor - tail),
                constant(0),
            )
            return ops.sub(count, dropped)

        def index_fn(index):
            *row, slot = index
            source_row = user_row(row)

            def tile(position):
                source_slot = FloorDiv(position, kv_factor)
                block = ops.masked(
                    ops.lt(
                        ops.index_expr(source_slot, torch.int32), constant(capacity)
                    ),
                    lambda: load_index([*source_row, source_slot]),
                    0,
                )
                offset = ModularIndexing(position, 1, kv_factor)
                return ops.add(
                    ops.mul(block, constant(kv_factor)),
                    ops.index_expr(offset, torch.int32),
                )

            value = tile(slot)
            if load_last_slot is not None:
                first_shifted = ops.add(
                    ops.mul(load_last_slot(row), constant(kv_factor)), constant(tail)
                )
                value = ops.where(
                    ops.lt(ops.index_expr(slot, torch.int32), first_shifted),
                    value,
                    tile(slot + kv_factor - tail),
                )
            # Slots past the count clamp to a valid block index.
            return ops.minimum(
                ops.maximum(value, constant(0)), constant(num_kv_blocks - 1)
            )

        new_counts = Pointwise.create(
            device=device, dtype=torch.int32, inner_fn=count_fn, ranges=rows
        )
        new_indices = Pointwise.create(
            device=device,
            dtype=torch.int32,
            inner_fn=index_fn,
            ranges=[*rows, capacity * kv_factor],
        )
        new_counts.realize()
        new_indices.realize()
        return new_counts, new_indices

    return (
        *expand(kv_num_blocks, kv_indices),
        *expand(full_kv_num_blocks, full_kv_indices),
    )


def _packed_decode_union_shape(
    query,
    key,
    kv_num_blocks,
    kv_indices,
    full_kv_num_blocks,
    full_kv_indices,
    sparse_kv_block_size,
) -> tuple[int, int] | None:
    """Return ``(Hkv, ceil(Sk / 128))`` if a per-Q-head decode mask must be merged."""
    if full_kv_num_blocks is None or full_kv_indices is None:
        return None
    nodes = (kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices)
    try:
        (_, hq, sq, _), (_, hkv, sk, _) = (_bhsd_sizes(node) for node in (query, key))
        block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
        shapes = [_syms(node.get_size()) for node in nodes]
        dtypes = [node.get_dtype() for node in nodes]
    except (AttributeError, TypeError, ValueError):
        return None
    if not (0 < sq < 128) or hkv <= 0 or hq == hkv or hq % hkv:
        return None
    num_kv_blocks = (sk + 127) // 128
    if block_size != 128 or sk <= 0 or num_kv_blocks > MAX_UNION_KV_BLOCKS:
        return None
    if any(dtype != torch.int32 for dtype in dtypes):
        return None
    count_shape, index_shape, full_count_shape, full_index_shape = shapes
    if (
        len(count_shape) != 3
        or count_shape[1:] != (hq, 1)
        or full_count_shape != count_shape
        or len(index_shape) != 4
        or index_shape[:3] != count_shape
        or len(full_index_shape) != 4
        or full_index_shape[:3] != count_shape
    ):
        return None
    return hkv, _ceil_div(_size_expr(key.get_size()[2]), 128)


def _forward_kv_splits(split_kv, query, key, head_dims) -> int:
    """``FLYDSL_SPLIT_KV``'s CTAs per decode CTA; 1 unless static, V in one launch."""
    if isinstance(split_kv, bool):
        if not split_kv:
            return 1
    elif not isinstance(split_kv, int) or split_kv < 1:
        raise ValueError("FlyDSL FLYDSL_SPLIT_KV must be a bool or a positive int")
    try:
        (batch, q_heads, seq_q, _), (_, kv_heads, seq_kv, _) = (
            _bhsd_sizes(node) for node in (query, key)
        )
    except (AttributeError, TypeError, ValueError):
        return 1
    if not all(isinstance(size, int) for size in (batch, seq_q, seq_kv)):
        return 1
    if kv_heads <= 0 or q_heads % kv_heads or head_dims is None:
        return 1
    target = _target_limits(query)
    if forward_value_slice(head_dims[0], head_dims[1], target) != head_dims[1]:
        return 1
    splits = forward_kv_splits(
        batch_size=batch,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        qk_head_dim=head_dims[0],
        v_head_dim=head_dims[1],
        requested=split_kv,
        target=target,
    )
    # The partial outputs are fp32.
    if splits * batch * q_heads * seq_q * head_dims[1] * 4 >= _MAX_BUFFER_BYTES:
        return 1
    return splits


def _combine_kv_splits(
    partials,
    split_logsumexp,
    split_max_scores,
    logsumexp,
    max_scores,
    layout,
    *,
    kv_splits,
    write_max_scores,
):
    """Merge a split-KV forward's [kv_splits * B, ...] partials into ``layout``."""
    batch, heads, seq_q, head_dim = map(V.graph.sizevars.guard_int, layout.size)

    def split_stride(node):
        stride = tuple(map(V.graph.sizevars.guard_int, node.get_stride()))
        return (batch * stride[0], *stride)

    input_nodes = [partials, split_logsumexp, split_max_scores, logsumexp, max_scores]
    choices: list[Any] = []
    appended, reason = _append_template_choice(
        flex_flydsl_split_kv_combine_template,
        choices,
        input_nodes=input_nodes,
        mutated_inputs=[logsumexp, max_scores],
        layout=layout,
        KV_SPLITS=kv_splits,
        BATCH_SIZE=batch,
        NUM_HEADS=heads,
        SEQ_Q=seq_q,
        HEAD_DIM=head_dim,
        PARTIAL_STRIDE=split_stride(partials),
        STATS_STRIDE=split_stride(split_logsumexp),
        O_STRIDE=tuple(map(V.graph.sizevars.guard_int, layout.stride)),
        LSE_STRIDE=tuple(map(V.graph.sizevars.guard_int, logsumexp.get_stride())),
        WRITE_MAX_SCORES=bool(write_max_scores),
        DTYPE=_FLYDSL_DTYPES[layout.dtype],
    )
    if not appended:
        raise RuntimeError(
            f"BACKEND='FLYDSL' could not register the split-KV combine: {reason}"
        )
    out, _ = autotune_select_algorithm(
        "flex_attention_flydsl_split_kv_combine", choices, input_nodes, layout
    )
    return out


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
    head_slow_order=True,
    split_kv=False,
    single_pass=False,
):
    """Lower the explicitly selected FlyDSL backend independently of Triton."""
    if not isinstance(head_slow_order, bool):
        raise ValueError("FlyDSL FLYDSL_HEAD_SLOW_ORDER must be a bool")
    if not isinstance(single_pass, bool):
        raise ValueError("FlyDSL FLYDSL_WIDE_SINGLE_PASS must be a bool")
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
    metadata = (kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices)
    *metadata, sparse_q_block_size, sparse_kv_block_size, block_states = (
        _maybe_split_block_mask(
            query, key, *metadata, sparse_q_block_size, sparse_kv_block_size
        )
    )
    union_shape = _packed_decode_union_shape(
        query,
        key,
        metadata[0],
        metadata[1],
        metadata[2],
        metadata[3],
        sparse_kv_block_size,
    )
    if union_shape is not None:
        num_kv_heads, num_kv_blocks = union_shape
        if block_states is None:
            # The union lists a block for every head of its group, so each head
            # keeps only the blocks its own lists hold.
            block_states = _BlockStates(
                _scatter_block_states(
                    *metadata,
                    state_shape=[
                        *map(_size_expr, metadata[0].get_size()),
                        num_kv_blocks,
                    ],
                    num_user_kv_blocks=num_kv_blocks,
                ),
                (128, 128),
            )
        metadata = _create_packed_decode_union_metadata(
            *metadata, num_kv_heads=num_kv_heads, num_kv_blocks=num_kv_blocks
        )
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = metadata
    score_mod_other_buffers, mask_mod_other_buffers, block_states = _prepare_captures(
        score_mod_other_buffers,
        mask_mod_other_buffers,
        block_states,
        query.get_device(),
    )
    mask_captures = _mask_captures(mask_mod_other_buffers, block_states)

    batch, heads, seq_q, _ = query.get_size()
    query_stride = query.get_stride()
    qk_dim, v_dim = (
        V.graph.sizevars.guard_int(node.get_size()[-1]) for node in (query, value)
    )
    padded_dims = kernel_head_dims(qk_dim, v_dim)
    kv_splits = _forward_kv_splits(split_kv, query, key, padded_dims)
    kv_head_dims = None
    qo_in_place = False
    out_width = v_dim
    if padded_dims is not None and padded_dims != (qk_dim, v_dim):
        key, value = (
            _require_bhsd_stride(node, broadcast_batch=True) for node in (key, value)
        )
        one_launch = (
            forward_value_slice(*padded_dims, _target_limits(query)) == padded_dims[1]
        )
        if one_launch and all(
            _reads_in_place(node, width)
            for node, width in zip((key, value), padded_dims, strict=True)
        ):
            kv_head_dims = (qk_dim, v_dim)
            query = _require_bhsd_stride(query)
            qo_in_place = (
                v_dim % 16 == 0
                and kv_splits == 1
                and _reads_in_place(query, padded_dims[0])
            )
        else:
            key = _pad_head_dim(key, padded_dims[0])
            value = _pad_head_dim(value, padded_dims[1])
        if not qo_in_place:
            query = _pad_head_dim(query, padded_dims[0])
            out_width = padded_dims[1]
    out_size = [batch, heads, seq_q, out_width]
    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        out_size,
        stride=infer_dense_strides(out_size, query_stride),
    )
    query = _require_bhsd_stride(query)
    key, value = (
        _require_bhsd_stride(node, broadcast_batch=True) for node in (key, value)
    )
    logsumexp = empty_strided(
        [batch, heads, seq_q], None, dtype=torch.float32, device=query.get_device()
    )
    max_scores = empty_strided(
        [batch, heads, seq_q], None, dtype=torch.float32, device=query.get_device()
    )
    merged_layout, merged_logsumexp, merged_max_scores = layout, logsumexp, max_scores
    if kv_splits > 1:
        split_size = [batch * kv_splits, heads, seq_q, out_size[3]]
        layout = FixedLayout(
            query.get_device(),
            torch.float32,
            split_size,
            stride=_contiguous_strides(split_size),
        )
        logsumexp, max_scores = (
            empty_strided(
                split_size[:3], None, dtype=torch.float32, device=query.get_device()
            )
            for _ in range(2)
        )
    score_mod_inputs = (
        []
        if is_trivial_score_graph(subgraph.graph_module)
        else _prepare_score_mod_captures(
            [b for b in score_mod_other_buffers if not isinstance(b, sympy.Expr)],
            query.get_device(),
        )
    )
    capture_count = len(mask_captures) + len(score_mod_inputs)
    capture_tables = (
        [_make_capture_table(capture_count, query.get_device())]
        if capture_count > MAX_DIRECT_CAPTURES
        else []
    )
    choices = []
    choice_kwargs = dict(
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
        subgraph_buffer=subgraph_buffer,
        score_mod_inputs=score_mod_inputs,
        block_states=block_states,
        head_slow_order=head_slow_order and block_mask_is_dense,
    )
    appended, reason = maybe_append_flydsl_flex_attention_choice(
        choices,
        **choice_kwargs,
        capture_table=next(iter(capture_tables), None),
        kv_head_dims=kv_head_dims,
        qo_in_place=qo_in_place,
        kv_splits=kv_splits,
        single_pass=single_pass,
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
            *mask_captures,
            *score_mod_inputs,
            *capture_tables,
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
    if kv_splits > 1:
        out = _combine_kv_splits(
            out,
            logsumexp,
            max_scores,
            merged_logsumexp,
            merged_max_scores,
            merged_layout,
            kv_splits=kv_splits,
            write_max_scores=write_max_scores,
        )
        logsumexp, max_scores = merged_logsumexp, merged_max_scores
    out = _unpad_head_dim(
        out, v_dim, infer_dense_strides([batch, heads, seq_q, v_dim], query_stride)
    )
    return out, logsumexp, max_scores
