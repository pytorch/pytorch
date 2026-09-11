"""ProxyTorchDispatchMode integration for custom_op: emit the FX graph node.

Node construction runs with proxy modes disabled so the tracer records only our
marker node and its getitems -- not the incidental ops metadata extraction would
otherwise re-trace (e.g. the empty_strided set_meta runs to fake a real output).
"""

from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    track_tensor_tree,
)

from .schema import (
    build_schema_body,
    flat_arg_names,
    mutated_flat_indices,
    schema_annotations,
)


if TYPE_CHECKING:
    from .custom_op import CustomOp


def tensor_metadata(t: torch.Tensor) -> tuple[Any, ...]:
    strides = tuple(t.stride()) if t.layout is torch.strided else None
    offset = t.storage_offset() if t.layout is torch.strided else None
    return tuple(t.shape), strides, offset, t.dtype, t.device, t.layout


def _run_and_build_overload(
    custom_op: "CustomOp",
    mode: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any, list[Any], list[Any]]:
    """Run the op and get/create the overload for this calling convention.

    Runs the real fn (real mode) or the registered fake kernel (fake mode),
    infers the leaf schema + aliasing from the observed values, and returns
    ``(out, op, flat_in, flat_out)`` for the caller to wire into the graph.
    """
    # Canonicalize the calling convention (positional vs keyword, defaults) so
    # logically-identical calls map to a single overload identity.
    bound = custom_op._sig.bind(*args, **kwargs)
    bound.apply_defaults()
    args, kwargs = bound.args, bound.kwargs

    flat_in, in_spec = pytree.tree_flatten((args, kwargs))
    arg_names = flat_arg_names(custom_op._sig, args, kwargs)
    mutated = mutated_flat_indices(custom_op._sig, custom_op._mutates, args, kwargs)

    orig_metadata = [
        tensor_metadata(x) if isinstance(x, torch.Tensor) else None for x in flat_in
    ]

    # Real mode runs the real fn; fake/symbolic mode requires a registered fake
    # kernel (no fallback -- the real fn may not be fake-traceable).
    if mode.tracing_mode == "real":
        impl = custom_op._fn
    elif custom_op._fake_fn is not None:
        impl = custom_op._fake_fn
    else:
        raise RuntimeError(
            f"{custom_op._ns}::{custom_op._name}: tracing_mode="
            f"{mode.tracing_mode!r} requires a fake kernel; register one with "
            f"@op.register_fake"
        )
    with disable_proxy_modes_tracing():
        out = impl(*args, **kwargs)

    for i, x in enumerate(flat_in):
        if isinstance(x, torch.Tensor) and tensor_metadata(x) != orig_metadata[i]:
            raise RuntimeError("metadata-changing input mutations are unsupported")

    # out_spec is None for a void op (see build_schema_body / _get_or_create_overload).
    flat_out, out_spec = ([], None) if out is None else pytree.tree_flatten(out)
    in_types, out_types = schema_annotations(flat_in, flat_out, mutated)
    schema_body = build_schema_body(arg_names, in_types, out_types)
    op = custom_op._get_or_create_overload(schema_body, in_spec, out_spec)
    return out, op, flat_in, flat_out


def trace_custom_op(
    custom_op: "CustomOp",
    mode: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    out, op, flat_in, flat_out = _run_and_build_overload(custom_op, mode, args, kwargs)

    tracer = mode.tracer
    with disable_proxy_modes_tracing():
        # Flat leaf args matching the leaf-only schema; the pytree structure is
        # metadata (op._in_spec), reconstructed only for printouts. name= keeps
        # the node readable (packet, not the mangled overload).
        proxy_leaves = tuple(tracer.unwrap_proxy(x) for x in flat_in)
        op_proxy = tracer.create_proxy(
            "call_function", op, proxy_leaves, {}, name=custom_op._name
        )

        if out is None:  # void op: node emitted, no outputs to track
            return None

        # track_tensor_tree emits getitems for tuple outputs and installs proxy
        # slots so downstream ops keep tracing.
        if len(flat_out) == 1:
            track_tensor_tree(flat_out[0], op_proxy, constant=None, tracer=tracer)
        else:
            track_tensor_tree(tuple(flat_out), op_proxy, constant=None, tracer=tracer)
    return out
