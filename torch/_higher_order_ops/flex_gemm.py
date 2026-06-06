# mypy: allow-untyped-defs
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _check_alias_and_mutation,
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


@dataclass(frozen=True)
class FlexGemmOpInfo:
    quack_name: str
    mat1_index: int
    mat2_index: int


FLEX_GEMM_OPS = {
    torch.ops.aten.mm.default: FlexGemmOpInfo("mm", 0, 1),
    torch.ops.aten.addmm.default: FlexGemmOpInfo("addmm", 1, 2),
}
_SUPPORTED_BACKENDS = {"TRITON", "QUACK"}


def supported_flex_gemm_op_names() -> str:
    return "/".join(op.name().removeprefix("aten::") for op in FLEX_GEMM_OPS)


_SUPPORTED_FLEX_GEMM_OP_NAMES = supported_flex_gemm_op_names()


def _normalize_flex_gemm_op(gemm_op: Callable[..., Any]) -> Callable[..., Any]:
    return {
        torch.mm: torch.ops.aten.mm.default,
        torch.addmm: torch.ops.aten.addmm.default,
    }.get(gemm_op, gemm_op)


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
        if gemm_op not in FLEX_GEMM_OPS:
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

        kernel_options = {"backend": "TRITON", **kernel_options}
        backend = kernel_options.get("backend", "TRITON")
        if backend not in _SUPPORTED_BACKENDS:
            raise RuntimeError(
                f"unsupported FlexGEMM backend: {backend}; "
                f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )
        return super().__call__(
            gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options
        )


_flex_gemm = FlexGemm()


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
    gemm_op = cast(torch._ops.OpOverload, _normalize_flex_gemm_op(gemm_op))

    def body_fn(*args: Any, **body_kwargs: Any) -> Any:
        return epilogue_fn(gemm_op(*args, **body_kwargs))

    body_fn._flex_gemm_accepts_kwargs = True  # type: ignore[attr-defined]
    return _flex_gemm(gemm_op, body_fn, gemm_args, gemm_kwargs, kernel_options)


def _body_accepts_kwargs(body_fn: Callable[..., Any], kwargs: Any) -> bool:
    if not kwargs or isinstance(body_fn, torch.fx.GraphModule):
        return False
    if getattr(body_fn, "_flex_gemm_accepts_kwargs", False):
        return True
    signature = inspect.signature(body_fn)
    return all(key in signature.parameters for key in kwargs)


def _call_flex_gemm_body(body_fn: Callable[..., Any], args: Any, kwargs: Any) -> Any:
    if _body_accepts_kwargs(body_fn, kwargs):
        return body_fn(*args, **kwargs)
    return body_fn(*args)


@_flex_gemm.py_impl(DispatchKey.CompositeExplicitAutograd)
def flex_gemm_dense(gemm_op, body_fn, args, kwargs, kernel_options):
    return _call_flex_gemm_body(body_fn, args, kwargs)


_flex_gemm.py_autograd_impl(autograd_not_implemented(_flex_gemm, deferred_error=True))


@_flex_gemm.py_impl(FakeTensorMode)
def flex_gemm_fake_tensor_mode(mode, gemm_op, body_fn, args, kwargs, kernel_options):
    with mode:
        return _call_flex_gemm_body(body_fn, tuple(args), {})


@_flex_gemm.py_functionalize_impl
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
            _flex_gemm(
                gemm_op,
                ctx.functionalize(body_fn),
                unwrapped_args,
                ctx.unwrap_tensors(kwargs),
                ctx.unwrap_tensors(kernel_options),
            )
        )


@_flex_gemm.py_impl(ProxyTorchDispatchMode)
def flex_gemm_proxy_torch_dispatch_mode(
    proxy_mode, gemm_op, body_fn, args, kwargs, kernel_options
):
    if proxy_mode.enable_tracing:
        flat_args = tuple(args)

        def tracing_body_fn(*flat_body_args):
            return _call_flex_gemm_body(body_fn, flat_body_args, kwargs)

        body_graph = reenter_make_fx(tracing_body_fn)(*flat_args)
        _, body_graph_name = unique_graph_id(proxy_mode, prefix="flex_gemm_body_graph")
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)
        proxy_args = pytree.tree_map(
            proxy_mode.tracer.unwrap_proxy,
            (gemm_op, body_graph, flat_args, kwargs, kernel_options),
        )
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function",
            _flex_gemm,
            proxy_args,
            {},
            name="flex_gemm",
        )
        return track_tensor_tree(
            _call_flex_gemm_body(body_fn, flat_args, kwargs),
            out_proxy,
            constant=None,
            tracer=proxy_mode.tracer,
        )
    return _flex_gemm(gemm_op, body_fn, args, kwargs, kernel_options)
