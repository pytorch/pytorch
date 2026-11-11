# mypy: allow-untyped-defs
"""Call into Helion for flexattention (BACKEND="HELION").

The forward and backward lower to the Helion kernels in ``flex_helion_kernels``,
compiled with Helion's CuTe backend into its fused tcgen05 flash kernels
(Blackwell GPUs). The score_mod, mask_mod and joint subgraphs go to the kernels
as FX GraphModules, which Helion inlines at trace time; Helion lowers each kernel
call through its own Inductor template lowering.

Configs: each kernel autotunes when compiled, at Helion's autotune effort
(``HELION_AUTOTUNE_EFFORT``; "none" takes Helion's default config), accepting
configs within ``kernel_options`` ``HELION_AUTOTUNE_TOLERANCE`` of the default
config. Helion caches tuned configs on disk.

The kernels are compiled for static shapes: the lowering specializes the graph
to its inputs' sizes, so a call at another shape recompiles.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE
from torch.utils._ordered_set import OrderedSet

from ... import config as inductor_config
from ...ir import (
    as_storage_and_layout,
    FixedLayout,
    ReinterpretView,
    TemplateBuffer,
    TensorBox,
)
from ...lowering import full, lowerings, to_dtype
from ...virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


aten = torch.ops.aten
prims = torch.ops.prims

# The fused kernels walk 128-wide KV blocks; "no BlockMask" (the frontend's one
# _LARGE_SPARSE_BLOCK_SIZE block) becomes dense tables of this block size.
_DENSE_BLOCK_SIZE = 128


@functools.lru_cache(maxsize=1)
def ensure_helion_available() -> bool:
    """Whether helion and the CuTe DSL (nvidia-cutlass-dsl) its CuTe backend
    compiles with are importable; cached."""
    # TODO: require the first Helion release after 1.5.2. 1.5.2 and older miss
    # flex fixes these kernels rely on (pytorch/helion#4296, #4298, #4319,
    # #4320, #4365, #4369); without them some score/mask mods compile to wrong
    # results under configs autotuning can pick, and max_scores cannot fuse.
    # They also lack helion.autotuner.AccuracyCheck (pytorch/helion#4380).
    try:
        return all(
            importlib.util.find_spec(m) is not None for m in ("helion", "cutlass")
        )
    except ImportError:
        return False


def _unavailable(reason: str) -> RuntimeError:
    return RuntimeError(
        f"BACKEND='HELION' is not supported here: {reason}. Use another backend."
    )


def _check_supported(
    query: TensorBox, key: TensorBox, value: TensorBox, scale: float
) -> None:
    if not ensure_helion_available():
        raise _unavailable("helion or nvidia-cutlass-dsl is not installed")
    device = query.get_device()
    if device is None or device.type != "cuda":
        raise _unavailable("it needs CUDA tensors")
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise _unavailable("its CuTe flex kernels need a Blackwell (sm100) GPU")
    dtype = query.get_dtype()
    if (
        key.get_dtype() != dtype
        or value.get_dtype() != dtype
        or dtype not in (torch.float16, torch.bfloat16)
    ):
        raise _unavailable(
            "query, key and value must share one dtype, float16 or bfloat16"
        )
    if not scale > 0:
        raise _unavailable(f"the scale must be positive (got {scale})")
    if query.get_size()[0] != key.get_size()[0]:
        raise _unavailable("NYI: query and key/value batch sizes differ")


def _specialized(node: TensorBox) -> TensorBox:
    """``node`` at static sizes and strides: symbolic ones are guarded to their
    current values, and the node is viewed at those."""
    guard = V.graph.sizevars.guard_int
    realized = TemplateBuffer.realize_template_input(node)
    layout = realized.get_layout()
    exprs = (*layout.size, *layout.stride, layout.offset)
    if all(isinstance(e, int) or e.is_number for e in exprs):
        return node
    storage, layout = as_storage_and_layout(realized)
    static = FixedLayout(
        layout.device,
        layout.dtype,
        [guard(s) for s in layout.size],
        [guard(s) for s in layout.stride],
        guard(layout.offset),
    )
    return TensorBox(ReinterpretView(data=storage, layout=static))


def _specialized_or_none(node: TensorBox | None) -> TensorBox | None:
    return None if node is None else _specialized(node)


def _captured(
    buffers: Sequence[Any], device: torch.device
) -> tuple[list[TensorBox], dict[int, int]]:
    """A mod's captured tensors at static sizes, and its captured integers
    (under dynamic shapes, the symbolic sizes it reads) by position, guarded
    to their values."""
    tensors = []
    ints = {}
    for i, buf in enumerate(buffers):
        if isinstance(buf, TensorBox):
            if buf.get_device() != device:
                raise _unavailable(
                    f"NYI: score_mod or mask_mod captures a tensor on "
                    f"{buf.get_device()}; move it to {device}, the device of "
                    "query, key and value"
                )
            tensors.append(_specialized(buf))
        elif isinstance(buf, int) or (isinstance(buf, sympy.Expr) and buf.is_integer):
            ints[i] = V.graph.sizevars.guard_int(buf)
        else:
            raise _unavailable(
                "NYI: score_mod or mask_mod captures a non-integer scalar; capture "
                "it as a tensor on device"
            )
    return tensors, ints


# -- the mods ------------------------------------------------------------------


class _ModTracer(torch.fx.Tracer):
    """The tracer FX rebuilds a prepared mod with when it is unpickled (as the
    Helion autotuner's benchmark subprocess loads the kernel args): it re-traces
    the mod's code, where ``hl.load`` must stay a call, not run on proxies."""

    def __init__(self) -> None:
        import helion.language as hl  # type: ignore[import-not-found]

        super().__init__(autowrap_functions=(hl.load,))


def _prepare_mod(
    gm: torch.fx.GraphModule, num_coords: int, captured_ints: Mapping[int, int]
) -> torch.fx.GraphModule:
    """A copy of a score_mod / mask_mod / score_mod_bwd subgraph a Helion kernel
    can inline: the placeholders after the ``num_coords`` leading ones are the
    captured values; the integers (``captured_ints``, by position) become
    constants and the tensors stay inputs, which a Helion device loop reads
    through ``hl.load``."""
    import helion.language as hl  # type: ignore[import-not-found]

    gm = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    graph = gm.graph
    captured = list(graph.find_nodes(op="placeholder"))[num_coords:]
    constants = {captured[i]: value for i, value in captured_ints.items()}
    if constants:
        for node in graph.nodes:
            node.args = torch.fx.node.map_arg(node.args, lambda n: constants.get(n, n))
            node.kwargs = torch.fx.node.map_arg(
                node.kwargs, lambda n: constants.get(n, n)
            )
        for node in constants:
            graph.erase_node(node)
    # The subgraphs were traced at the input dtype; the kernels evaluate them
    # in fp32, as eager does, so a cast between float dtypes only adds rounding
    # (and in the joint graph it splits the region the fused backward matches).
    # Inductor elides such casts too, unless emulate_precision_casts asks to
    # keep them.
    if not inductor_config.emulate_precision_casts:
        for node in graph.find_nodes(
            op="call_function", target=prims.convert_element_type.default
        ):
            src, dtype = node.args[:2]
            src_val = src.meta.get("val") if isinstance(src, torch.fx.Node) else None
            if (
                dtype.is_floating_point
                and isinstance(src_val, torch.Tensor)
                and src_val.dtype.is_floating_point
            ):
                node.replace_all_uses_with(src)
                graph.erase_node(node)
    buffers = OrderedSet(list(graph.find_nodes(op="placeholder"))[num_coords:])
    for node in graph.nodes:
        if node.target is aten.index.Tensor and node.args[0] in buffers:
            indices = list(node.args[1])
            if any(i is None for i in indices):
                raise _unavailable(
                    "NYI: score_mod or mask_mod indexes a captured tensor with a slice"
                )
            node.target = hl.load
            node.args = (node.args[0], indices)
    for buf in buffers:
        for user in buf.users:
            if user.target is not hl.load:
                raise _unavailable(
                    "NYI: score_mod or mask_mod uses a captured tensor other than "
                    "by indexing it with the coordinates"
                )
    gm.recompile()
    gm._tracer_cls = _ModTracer
    return gm


def _score_mod_bwd(
    joint_gm: torch.fx.GraphModule, captured_ints: Mapping[int, int]
) -> torch.fx.GraphModule:
    """Helion's ``score_mod_bwd(grad, score, b, h, q_idx, kv_idx, *buffers)`` from
    the joint graph ``(score, b, h, m, n, grad_score_mod, *buffers) ->
    (grad_score, grad_b, ..., *buffer_grads)``."""
    gm = torch.fx.GraphModule(joint_gm, copy.deepcopy(joint_gm.graph))
    graph = gm.graph
    placeholders = list(graph.find_nodes(op="placeholder"))
    placeholders[0].prepend(placeholders[5])
    (output,) = graph.find_nodes(op="output")
    outs = output.args[0]
    if any(o is not None for o in outs[5:]):
        raise _unavailable("NYI: gradients of tensors captured by score_mod")
    output.args = (outs[0],)
    graph.eliminate_dead_code()
    gm.recompile()
    return _prepare_mod(gm, 6, captured_ints)


# -- BlockMask tables ------------------------------------------------------------


def _unpack_block_mask(
    block_mask: tuple[Any, ...],
) -> tuple[list[Any], list[Any], Any, Any, Any]:
    """The KV-side tables (num_blocks, indices, full_num_blocks, full_indices),
    the Q-side tables, the sparse Q and KV block sizes and the mask graph of
    flex_attention's ``block_mask`` argument."""
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
        _,  # dq_write_order (Triton backward only)
        _,  # dq_write_order_full
        _,  # dq_kv_order
        _,  # dq_kv_order_spt
        sparse_q_block_size,
        sparse_kv_block_size,
        mask_graph,
    ) = block_mask
    return (
        [kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices],
        [q_num_blocks, q_indices, full_q_num_blocks, full_q_indices],
        sparse_q_block_size,
        sparse_kv_block_size,
        mask_graph,
    )


def _realized(node: TensorBox) -> TensorBox:
    node.realize()
    return node


def _dense_tables(rows: int, cols: int, device: torch.device) -> list[TensorBox]:
    """(num_blocks, indices, full_num_blocks, full_indices) of a mask whose
    ``_DENSE_BLOCK_SIZE`` blocks are all full."""
    nrows = -(-rows // _DENSE_BLOCK_SIZE)
    ncols = -(-cols // _DENSE_BLOCK_SIZE)
    i32 = torch.int32
    iota = lowerings[prims.iota.default](
        ncols, start=0, step=1, dtype=i32, device=device, requires_grad=False
    )
    full_indices = lowerings[aten.clone.default](
        lowerings[aten.expand.default](
            lowerings[aten.view.default](iota, [1, 1, 1, ncols]), [1, 1, nrows, ncols]
        )
    )
    return [
        _realized(full([1, 1, nrows], 0, dtype=i32, device=device)),
        _realized(full([1, 1, nrows, ncols], 0, dtype=i32, device=device)),
        _realized(full([1, 1, nrows], ncols, dtype=i32, device=device)),
        _realized(full_indices),
    ]


def _block_tables(
    tables: Sequence[TensorBox | None],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    rows: int,
    cols: int,
    device: torch.device,
) -> tuple[list[TensorBox], int, int]:
    """The four (num_blocks, indices, full_num_blocks, full_indices) tables of
    one walk direction and its (row, column) block sizes, with full-block
    tables for a BlockMask without full blocks and dense tables for none."""
    if sparse_q_block_size == sparse_kv_block_size == _LARGE_SPARSE_BLOCK_SIZE:
        return _dense_tables(rows, cols, device), _DENSE_BLOCK_SIZE, _DENSE_BLOCK_SIZE
    num_blocks, indices, full_num_blocks, full_indices = tables
    if num_blocks is None or indices is None:
        raise _unavailable("NYI: a BlockMask built with compute_q_blocks=False")
    if full_num_blocks is None or full_indices is None:
        i32 = torch.int32
        full_num_blocks = _realized(
            full([int(s) for s in num_blocks.get_size()], 0, dtype=i32, device=device)
        )
        full_indices = _realized(
            full([int(s) for s in indices.get_size()], 0, dtype=i32, device=device)
        )
    return (
        [num_blocks, indices, full_num_blocks, full_indices],
        sparse_q_block_size,
        sparse_kv_block_size,
    )


# -- kernel selection and lowering ------------------------------------------------


def _fake(node: TensorBox) -> torch.Tensor:
    realized = TemplateBuffer.realize_template_input(node)
    with V.fake_mode:
        return torch.empty_strided(
            [int(s) for s in realized.get_size()],
            [int(s) for s in realized.get_stride()],
            dtype=realized.get_dtype(),
            device=realized.get_device(),
        )


def _example(node: TensorBox) -> torch.Tensor:
    """A real input like ``node`` for autotuning: random floats, zero integers
    (a captured integer tensor may index another)."""
    realized = TemplateBuffer.realize_template_input(node)
    size = [int(s) for s in realized.get_size()]
    stride = [int(s) for s in realized.get_stride()]
    dtype = realized.get_dtype()
    out = torch.empty_strided(size, stride, dtype=dtype, device=realized.get_device())
    return out.normal_() if dtype.is_floating_point else out.zero_()


def _dense_example_tables(
    tables: Sequence[TensorBox],
) -> list[torch.Tensor]:
    """Real BlockMask tables shaped like ``tables`` with every block full, for
    autotuning (Inductor's random example inputs would index out of range)."""
    num_blocks, indices, full_num_blocks, full_indices = (_example(t) for t in tables)
    num_blocks.zero_()
    indices.zero_()
    ncols = full_indices.size(-1)
    full_num_blocks.fill_(ncols)
    full_indices.copy_(
        torch.arange(ncols, dtype=full_indices.dtype, device=full_indices.device)
    )
    return [num_blocks, indices, full_num_blocks, full_indices]


def _container_specs(
    tensor_args: dict[str, TensorBox],
    constant_args: dict[str, object],
    score_mod_buffers: Sequence[TensorBox],
    mask_mod_buffers: Sequence[TensorBox],
) -> None:
    """Add the captured tensors as the kernels' tuple params, flattened like any
    Helion container arg (``name.i`` tensor args plus a treespec)."""
    import torch.utils._pytree as pytree

    specs = {}
    for name, bufs in (
        ("score_mod_buffers", score_mod_buffers),
        ("mask_mod_buffers", mask_mod_buffers),
    ):
        if not bufs:
            constant_args[name] = ()
            continue
        for i, buf in enumerate(bufs):
            tensor_args[f"{name}.{i}"] = buf
        specs[name] = pytree.treespec_dumps(pytree.tree_flatten(tuple(bufs))[1])
    if specs:
        constant_args["__container_specs"] = specs


def _kernel_args(
    kernel: Any,
    tensor_args: dict[str, Any],
    constant_args: dict[str, object],
    leaf: Callable[[TensorBox], torch.Tensor],
) -> tuple[object, ...]:
    from helion._compiler._dynamo.higher_order_ops import (  # type: ignore[import-not-found]
        _rebuild_container_args,
    )

    all_args = {**constant_args, **{n: leaf(t) for n, t in tensor_args.items()}}
    _rebuild_container_args(all_args)
    return tuple(all_args[n] for n in kernel.signature.parameters)


def _select_kernel(
    direction: str,
    kernel_options: dict[str, Any],
    fake_args: tuple[object, ...],
    tuning_args: Callable[[], tuple[object, ...]],
) -> Any:
    """The kernel to lower, fixed to the best config autotuning finds (Helion's
    default config at ``HELION_AUTOTUNE_EFFORT=none``)."""
    from . import flex_helion_kernels

    tolerance = kernel_options.get(
        "HELION_AUTOTUNE_TOLERANCE", flex_helion_kernels.AUTOTUNE_TOLERANCE
    )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not tolerance > 0
    ):
        raise ValueError(
            f"Invalid HELION_AUTOTUNE_TOLERANCE {tolerance!r}; must be a positive number"
        )
    kernel = flex_helion_kernels.get_kernel(direction, tolerance=float(tolerance))
    bound = kernel._bind_isolated(fake_args)
    knob = "cute_flash_flex_" if direction == "forward" else "cute_flash_bwd_"
    if not any(k.startswith(knob) for k in bound.config_spec.default_config().config):
        matcher = "flex_match" if direction == "forward" else "flex_bwd_match"
        hint = ""
        if direction == "backward" and inductor_config.emulate_precision_casts:
            hint = (
                "; NYI: emulate_precision_casts=True keeps the float casts in "
                "score_mod's gradient, which the fused backward does not fit"
            )
        # TODO: put the matcher's decline reason (logged at DEBUG as
        # "<label> declined: <reason>") in this error.
        raise _unavailable(
            f"this flex_attention {direction} does not fit Helion's fused CuTe "
            f"flash kernel; HELION_LOGS=+helion._compiler.cute.{matcher} logs "
            f"why{hint}"
        )
    if kernel.settings.autotune_effort == "none":
        return kernel
    config = kernel.autotune(tuning_args(), force=False)
    return flex_helion_kernels.get_kernel(direction, config=dict(config.config))


def _lower(
    direction: str,
    kernel_options: dict[str, Any],
    tensor_args: dict[str, TensorBox],
    constant_args: dict[str, object],
    tables: Sequence[TensorBox],
) -> tuple[TensorBox, ...]:
    """Lower one kernel call through Helion's Inductor template lowering."""
    from helion._compiler._dynamo.higher_order_ops import (  # type: ignore[import-not-found]
        helion_kernel_side_table,
    )
    from helion._compiler._dynamo.variables import (  # type: ignore[import-not-found]
        infer_output_spec,
    )
    from helion._compiler._inductor.template_buffer import (  # type: ignore[import-not-found]
        lower_helion_kernel,
    )

    from . import flex_helion_kernels

    signature_kernel = flex_helion_kernels.get_kernel(direction)
    fake_args = _kernel_args(signature_kernel, tensor_args, constant_args, _fake)

    def tuning_args() -> tuple[object, ...]:
        examples = dict(
            zip((t.get_name() for t in tables), _dense_example_tables(tables))
        )

        def leaf(node: TensorBox) -> torch.Tensor:
            example = examples.get(node.get_name())
            return _example(node) if example is None else example

        return _kernel_args(signature_kernel, tensor_args, constant_args, leaf)

    kernel = _select_kernel(direction, kernel_options, fake_args, tuning_args)
    return lower_helion_kernel(
        kernel_idx=helion_kernel_side_table.add_kernel(kernel),
        constant_args=constant_args,
        tensor_args=tensor_args,
        output_spec=infer_output_spec(kernel, fake_args),
    )


def create_flex_helion_attention_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    subgraph: Any,
    block_mask: tuple[Any, ...],
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: Sequence[Any],
    mask_mod_other_buffers: Sequence[Any],
) -> tuple[TensorBox, TensorBox, TensorBox]:
    """Lower the flex_attention forward to the Helion forward kernel.

    Returns (out, logsumexp in base 2, max_scores in base 2); max_scores is an
    empty placeholder unless the frontend asks for it (``OUTPUT_MAX``).
    """
    kv_tables, _, sparse_q, sparse_kv, mask_graph = _unpack_block_mask(block_mask)
    query, key, value = (_specialized(t) for t in (query, key, value))
    _check_supported(query, key, value, scale)
    kv_tables = [_specialized_or_none(t) for t in kv_tables]
    device = query.get_device_or_error()
    score_mod_buffers, score_mod_ints = _captured(score_mod_other_buffers, device)
    mask_mod_buffers, mask_mod_ints = _captured(mask_mod_other_buffers, device)
    tables, block_m, block_n = _block_tables(
        kv_tables,
        V.graph.sizevars.guard_int(sparse_q),
        V.graph.sizevars.guard_int(sparse_kv),
        int(query.get_size()[2]),
        int(key.get_size()[2]),
        device,
    )
    tensor_args: dict[str, TensorBox] = {
        "query": query,
        "key": key,
        "value": value,
        "block_mask_kv_num_blocks": tables[0],
        "block_mask_kv_indices": tables[1],
        "block_mask_full_kv_num_blocks": tables[2],
        "block_mask_full_kv_indices": tables[3],
    }
    constant_args: dict[str, object] = {
        "score_mod": _prepare_mod(subgraph.graph_module, 5, score_mod_ints),
        "block_mask_mask_mod": _prepare_mod(mask_graph.graph_module, 4, mask_mod_ints),
        "block_mask_m": block_m,
        "block_mask_n": block_n,
        "scale": float(scale),
        # A row-max store keeps the fused plan off its dense bodies, so the
        # kernel stores it only on request.
        "output_max": bool(kernel_options.get("OUTPUT_MAX", False)),
    }
    _container_specs(tensor_args, constant_args, score_mod_buffers, mask_mod_buffers)

    out, lse, max_scores = _lower(
        "forward",
        kernel_options,
        tensor_args,
        constant_args,
        tables,
    )
    return out, lse, max_scores


def create_flex_helion_attention_backward_kernel(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    out: TensorBox,
    logsumexp: TensorBox,
    grad_out: TensorBox,
    grad_logsumexp: TensorBox | None,
    fw_graph: Any,
    joint_graph: Any,
    block_mask: tuple[Any, ...],
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: Sequence[Any],
    mask_mod_other_buffers: Sequence[Any],
) -> tuple[TensorBox, TensorBox, TensorBox, tuple[None, ...]]:
    """Lower the flex_attention backward to the Helion backward kernel; returns
    (grad_query, grad_key, grad_value, captured-tensor grads)."""
    _, q_tables, sparse_q, sparse_kv, mask_graph = _unpack_block_mask(block_mask)
    query, key, value, out, logsumexp, grad_out = (
        _specialized(t) for t in (query, key, value, out, logsumexp, grad_out)
    )
    _check_supported(query, key, value, scale)
    grad_logsumexp = _specialized_or_none(grad_logsumexp)
    q_tables = [_specialized_or_none(t) for t in q_tables]
    device = query.get_device_or_error()
    score_mod_buffers, score_mod_ints = _captured(score_mod_other_buffers, device)
    mask_mod_buffers, mask_mod_ints = _captured(mask_mod_other_buffers, device)
    # The Q-side tables list Q blocks per KV block: rows are KV blocks.
    tables, block_m, block_n = _block_tables(
        q_tables,
        V.graph.sizevars.guard_int(sparse_q),
        V.graph.sizevars.guard_int(sparse_kv),
        int(key.get_size()[2]),
        int(query.get_size()[2]),
        device,
    )
    # The softmax-backward row terms: delta = rowsum(out * grad_out), and the
    # forward's base-2 logsumexp with no-visible-key rows mapped to 0.
    f32 = torch.float32
    delta = lowerings[aten.sum.dim_IntList](
        lowerings[aten.mul.Tensor](to_dtype(out, f32), to_dtype(grad_out, f32)), [-1]
    )
    if grad_logsumexp is not None:
        # d lse / d score = P, so the logsumexp's gradient adds P * grad_lse to
        # dS = P * (dP - delta): fold it into delta. The forward's logsumexp is
        # base 2, so its gradient in natural units is grad_logsumexp * log2(e).
        delta = lowerings[aten.sub.Tensor](
            delta, lowerings[aten.mul.Tensor](grad_logsumexp, 1.0 / math.log(2.0))
        )
    lse2 = lowerings[aten.where.self](
        lowerings[aten.eq.Scalar](logsumexp, float("-inf")), 0.0, logsumexp
    )
    tensor_args: dict[str, TensorBox] = {
        "query": query,
        "key": key,
        "value": value,
        "grad_out": grad_out,
        "lse2": lse2,
        "delta": delta,
        "block_mask_q_num_blocks": tables[0],
        "block_mask_q_indices": tables[1],
        "block_mask_full_q_num_blocks": tables[2],
        "block_mask_full_q_indices": tables[3],
    }
    constant_args: dict[str, object] = {
        "score_mod": _prepare_mod(fw_graph.graph_module, 5, score_mod_ints),
        "score_mod_bwd": _score_mod_bwd(joint_graph.graph_module, score_mod_ints),
        "block_mask_mask_mod": _prepare_mod(mask_graph.graph_module, 4, mask_mod_ints),
        "block_mask_m": block_m,
        "block_mask_n": block_n,
        "scale": float(scale),
    }
    _container_specs(tensor_args, constant_args, score_mod_buffers, mask_mod_buffers)

    dq, dk, dv = _lower(
        "backward",
        kernel_options,
        tensor_args,
        constant_args,
        tables,
    )
    captured_grads = tuple(None for _ in score_mod_other_buffers)
    return to_dtype(dq, query.get_dtype()), dk, dv, captured_grads


# -- FX graph cache ---------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _helion_source_hash() -> str:
    import helion  # type: ignore[import-not-found]

    root = Path(helion.__file__).parent
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def helion_cache_key() -> str:
    """What the generated code of a BACKEND="HELION" graph depends on beyond the
    graph and torch's own source: Helion's source, the CuTe DSL it compiles
    with, and Helion's environment settings."""
    versions: dict[str, str | None] = {}
    for dist in ("helion", "nvidia-cutlass-dsl"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    settings = sorted(
        (name, value)
        for name, value in os.environ.items()
        if name.startswith("HELION_")
    )
    return json.dumps([_helion_source_hash(), versions, settings])
