import builtins
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import get_proxy_slot, ProxyTorchDispatchMode


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


@print.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def print_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode, format_str: str, **kwargs: object
) -> None:
    def _unwrap_proxy(e: tuple) -> Any:
        if not isinstance(e, (torch.Tensor, torch.SymInt, torch.SymFloat)):
            return e
        return get_proxy_slot(
            cast(torch.Tensor, e),
            mode.tracer,
            e,
            lambda e: e.proxy,  # type: ignore[attr-defined]
        )

    node_args = (format_str, kwargs)
    proxy_args = pytree.tree_map(_unwrap_proxy, node_args)
    mode.tracer.create_proxy("call_function", print, proxy_args, {}, name="print")

@print.py_functionalize_impl
# pyre-ignore
def print_functionalize(ctx: Any, format_str: str, **kwargs: object) -> None:
    with ctx.redispatch_to_next():
        return None

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


print.fallthrough(torch._C.DispatchKey.AutogradCPU)
print.fallthrough(torch._C.DispatchKey.AutogradCUDA)
