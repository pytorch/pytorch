import builtins

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_slot,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

from typing import Any, cast


class Print(HigherOrderOperator):
    """
    print(format_str, **kwargs) -> None

    This Higher Order Operator (HOP) provides a functional version of print for use in PyTorch graphs.
    It enables format printing with named arguments, e.g., torch._higher_order_ops.print("moo {x} {y}", x=1, y=2).

    This HOP enables printing without causing graph break.
    """

    def __init__(self) -> None:
        super().__init__("print")

    def __call__(self, format_str: str, **kwargs: object) -> object:
        assert isinstance(format_str, str)
        return super().__call__(format_str, **kwargs)


print = Print()


def trace_print(proxy_mode, func_overload, format_str, **kwargs):
    def _unwrap_proxy(e):
        if not isinstance(e, (torch.Tensor, torch.SymInt, torch.SymFloat)):
            return e
        return get_proxy_slot(
            cast(torch.Tensor, e),
            proxy_mode.tracer,
            e,
            lambda e: e.proxy,  # type: ignore[attr-defined]
        )

    if not isinstance(format_str, str):
        raise ValueError("print's first argument must be a string")

    node_args = (format_str, kwargs)
    proxy_args = pytree.tree_map(_unwrap_proxy, node_args)
    return proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="print"
    )

@print.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def print_proxy_torch_dispatch_mode(mode, format_str, **kwargs):
    res = trace_print(mode, print, format_str, **kwargs)
    return res

@print.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def print_cpu(format_str: str, **kwargs: object) -> None:
    # Ensure all immutable_dict/list in kwargs are converted to regular dict/list
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_kwargs = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        kwargs,
        lambda a: isinstance(a, tuple(map_types.keys())),
      )
    # Use built-in print to avoid recursion with the HOP print
    builtins.print(format_str.format(**new_kwargs))
