# mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


_SUPPORTED_GEMM_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_mm.default,
}
_SUPPORTED_BACKENDS = {"TRITON", "CUTEDSL", "QUACK"}


class GemmEpilogueFusion(HigherOrderOperator):
    def __init__(self):
        super().__init__("gemm_epilogue_fusion")

    def __call__(
        self,
        gemm_op: torch._ops.OpOverload,
        body_fn: Callable[..., Any],
        gemm_args: tuple[Any, ...],
        gemm_kwargs: dict[str, Any],
        kernel_options: dict[str, Any],
    ):
        if gemm_op not in _SUPPORTED_GEMM_OPS:
            raise RuntimeError(f"unsupported GEMM op for epilogue fusion: {gemm_op}")

        if not isinstance(gemm_args, tuple):
            gemm_args = tuple(gemm_args)

        if not all(
            isinstance(
                t,
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
            for t in gemm_args
        ):
            raise RuntimeError(
                "gemm_args must be a tuple of tensors, SymInts, SymFloats, "
                f"SymBools, ints, floats, or bools, got {gemm_args}"
            )

        if not isinstance(gemm_kwargs, dict):
            raise RuntimeError(f"gemm_kwargs must be a dict, got {type(gemm_kwargs)}")

        if not isinstance(kernel_options, dict):
            raise RuntimeError(
                f"kernel_options must be a dict, got {type(kernel_options)}"
            )

        backend = kernel_options.get("backend", "TRITON")
        if backend not in _SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"unsupported GEMM epilogue backend: {backend}; "
                f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )

        return super().__call__(
            gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
        )


_gemm_epilogue_fusion = GemmEpilogueFusion()


def gemm_epilogue_fusion(
    gemm_op: torch._ops.OpOverload,
    gemm_args: tuple[Any, ...],
    epilogue_fn: Callable[[Any], Any],
    *,
    gemm_kwargs: dict[str, Any] | None = None,
    kernel_options: dict[str, Any] | None = None,
):
    if gemm_kwargs is None:
        gemm_kwargs = {}
    if kernel_options is None:
        kernel_options = {"backend": "TRITON"}

    def body_fn(*args):
        return epilogue_fn(gemm_op(*args, **gemm_kwargs))

    return _gemm_epilogue_fusion(
        gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
    )


@_gemm_epilogue_fusion.py_impl(DispatchKey.CompositeExplicitAutograd)
def gemm_epilogue_fusion_dense(gemm_op, body_fn, args, kwargs, kernel_options):
    return body_fn(*args, **kwargs)


_gemm_epilogue_fusion.py_autograd_impl(
    autograd_not_implemented(_gemm_epilogue_fusion, deferred_error=True)
)


@_gemm_epilogue_fusion.py_impl(FakeTensorMode)
def gemm_epilogue_fusion_fake_tensor_mode(
    mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    flat_args = pytree.tree_leaves(args)
    with mode:
        return body_fn(*flat_args, **kwargs)


@_gemm_epilogue_fusion.py_functionalize_impl
def gemm_epilogue_fusion_functionalize(
    ctx, gemm_op, body_fn, args, kwargs, kernel_options
):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    unwrapped_kernel_options = ctx.unwrap_tensors(kernel_options)
    with ctx.redispatch_to_next():
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(
            body_fn, unwrapped_args, "gemm_epilogue_fusion", pre_dispatch
        )

        outputs = _gemm_epilogue_fusion(
            gemm_op,
            functional_body_fn,
            unwrapped_args,
            unwrapped_kwargs,
            unwrapped_kernel_options,
        )
        return ctx.wrap_tensors(outputs)


@_gemm_epilogue_fusion.py_impl(ProxyTorchDispatchMode)
def gemm_epilogue_fusion_proxy_torch_dispatch_mode(
    proxy_mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    if proxy_mode.enable_tracing:
        flat_args = tuple(pytree.tree_leaves(args))
        body_graph = reenter_make_fx(body_fn)(*flat_args, **kwargs)
        _, body_graph_name = unique_graph_id(
            proxy_mode, prefix="gemm_epilogue_fusion_body_graph"
        )
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)
        proxy_args = pytree.tree_map(
            proxy_mode.tracer.unwrap_proxy,
            (gemm_op, body_graph, flat_args, kwargs, kernel_options),
        )
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function",
            _gemm_epilogue_fusion,
            proxy_args,
            {},
            name="gemm_epilogue_fusion",
        )
        out = body_fn(*flat_args, **kwargs)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )
    return _gemm_epilogue_fusion(gemm_op, body_fn, args, kwargs, kernel_options)
