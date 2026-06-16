# mypy: allow-untyped-defs
import dataclasses
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _check_alias_and_mutation,
    autograd_not_implemented,
    reenter_make_fx,
    register_fake,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


@dataclasses.dataclass(frozen=True)
class FlexGemmOpSpec:
    """Canonical FlexGEMM view of a supported GEMM op's operand layout."""

    name: str
    mat1_index: int
    mat2_index: int
    bias_index: int | None = None


FLEX_GEMM_OP_SPECS = {
    torch.ops.aten.mm.default: FlexGemmOpSpec("mm", 0, 1),
    torch.ops.aten.addmm.default: FlexGemmOpSpec("addmm", 1, 2, bias_index=0),
}
FLEX_GEMM_OP_ALIASES = {
    torch.mm: torch.ops.aten.mm.default,
    torch.addmm: torch.ops.aten.addmm.default,
}
_SUPPORTED_BACKENDS = {"TRITON", "QUACK"}


_SUPPORTED_FLEX_GEMM_OP_NAMES = "/".join(
    spec.name for spec in FLEX_GEMM_OP_SPECS.values()
)


class FlexGemm(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_gemm")

    def __call__(
        self,
        gemm_op: torch._ops.OpOverload,
        body_fn: Callable[..., Any],
        gemm_args: tuple[Any, ...],
        gemm_kwargs: dict[str, Any],
        kernel_options: dict[str, Any],
    ) -> Any:
        if gemm_op not in FLEX_GEMM_OP_SPECS:
            raise RuntimeError(f"unsupported GEMM op for FlexGEMM: {gemm_op}")
        if not isinstance(gemm_args, (tuple, list)):
            raise RuntimeError(f"gemm_args must be a tuple/list, got {type(gemm_args)}")
        gemm_args = tuple(gemm_args)
        if not all(
            isinstance(
                arg,
                (
                    torch.Tensor,
                    torch.SymInt,
                    torch.SymFloat,
                    torch.SymBool,
                    int,
                    float,
                    bool,
                ),
            )
            for arg in gemm_args
        ):
            raise RuntimeError(
                "gemm_args must be a tuple of tensors, SymInts, SymFloats, "
                f"SymBools, ints, floats, or bools, got {gemm_args}"
            )
        if not isinstance(gemm_kwargs, dict):
            raise RuntimeError(f"gemm_kwargs must be a dict, got {type(gemm_kwargs)}")
        if any(
            isinstance(value, torch.Tensor) for value in pytree.tree_leaves(gemm_kwargs)
        ):
            raise RuntimeError(
                "gemm_kwargs must not contain tensor values; pass tensor inputs through gemm_args"
            )
        if not isinstance(kernel_options, dict):
            raise RuntimeError(
                f"kernel_options must be a dict, got {type(kernel_options)}"
            )

        kernel_options = dict(kernel_options)
        kernel_options.setdefault("backend", "TRITON")
        backend = kernel_options["backend"]
        if backend not in _SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"unsupported FlexGEMM backend: {backend}; "
                f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )
        return super().__call__(
            gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
        )


flex_gemm_hop = FlexGemm()


def flex_gemm(
    gemm_op: Callable[..., Any],
    gemm_args: tuple[Any, ...],
    epilogue_fn: Callable[[Any], Any],
    *,
    gemm_kwargs: dict[str, Any] | None = None,
    kernel_options: dict[str, Any] | None = None,
) -> Any:
    if gemm_kwargs is None:
        gemm_kwargs = {}
    if kernel_options is None:
        kernel_options = {}
    gemm_op = cast(torch._ops.OpOverload, FLEX_GEMM_OP_ALIASES.get(gemm_op, gemm_op))

    def body_fn(*args: Any) -> Any:
        # Keep the traced body positional-only; the HOP carries gemm_kwargs for lowering.
        return epilogue_fn(gemm_op(*args, **gemm_kwargs))

    return flex_gemm_hop(gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options)


@flex_gemm_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def flex_gemm_dense(gemm_op, body_fn, args, kwargs, kernel_options):
    return body_fn(*args)


flex_gemm_hop.py_autograd_impl(
    autograd_not_implemented(flex_gemm_hop, deferred_error=True)
)


@register_fake(flex_gemm_hop)
def flex_gemm_fake_tensor_mode(gemm_op, body_fn, args, kwargs, kernel_options):
    return body_fn(*args)


@flex_gemm_hop.py_functionalize_impl
def flex_gemm_functionalize(ctx, gemm_op, body_fn, args, kwargs, kernel_options):
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        _check_alias_and_mutation(
            body_fn,
            unwrapped_args,
            "flex_gemm",
            hasattr(ctx, "mode") and ctx.mode.pre_dispatch,
        )
        return ctx.wrap_tensors(
            flex_gemm_hop(
                gemm_op,
                ctx.functionalize(body_fn),
                unwrapped_args,
                ctx.unwrap_tensors(kwargs),
                ctx.unwrap_tensors(kernel_options),
            )
        )


@flex_gemm_hop.py_impl(ProxyTorchDispatchMode)
def flex_gemm_proxy_torch_dispatch_mode(
    proxy_mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    if proxy_mode.enable_tracing:
        flat_args = tuple(args)

        def tracing_body_fn(*flat_body_args):
            return body_fn(*flat_body_args)

        body_graph = reenter_make_fx(tracing_body_fn)(*flat_args)
        _, body_graph_name = unique_graph_id(proxy_mode, prefix="flex_gemm_body_graph")
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)
        proxy_args = pytree.tree_map(
            proxy_mode.tracer.unwrap_proxy,
            (gemm_op, body_graph, flat_args, kwargs, kernel_options),
        )
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function",
            flex_gemm_hop,
            proxy_args,
            {},
            name="flex_gemm",
        )
        return track_tensor_tree(
            body_fn(*flat_args),
            out_proxy,
            constant=None,
            tracer=proxy_mode.tracer,
        )
    return flex_gemm_hop(gemm_op, body_fn, args, kwargs, kernel_options)
