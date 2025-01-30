# mypy: allow-untyped-defs
from dataclasses import dataclass
from typing import Callable

import torch
import torch.fx.node
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


def is_graphable(val) -> bool:
    return isinstance(val, torch.fx.node.base_types)


def flatten(stuff):
    # Flatten everything
    flat_args, spec = pytree.tree_flatten(stuff)
    for arg in flat_args:
        # TODO: better error message
        assert is_graphable(arg), f"Expected graphable, got {type(arg)}"
    return flat_args, spec


def unflatten(flat_args, spec):
    stuff = pytree.tree_unflatten(flat_args, spec)
    return stuff


@dataclass
class ConstantFunction:
    func: Callable


pytree.register_constant(ConstantFunction)


def plop_in_graph(f):
    def inner(*args, **kwargs):
        flat_args, in_spec = flatten((args, kwargs))
        _, f_spec = pytree.tree_flatten(ConstantFunction(f))
        return flat_apply(f_spec, in_spec, *flat_args)

    return inner


class FlatApply(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_apply")

    def __call__(self, func_spec, in_spec, *args):
        """
        The semantics of flat_apply(func_spec, in_spec, *args) is the following:

        >>> func = pytree.tree_unflatten([], func_spec)
        >>> args, kwargs = unflatten(args, in_spec)
        >>> output = func(*args, **kwargs)
        >>> return output

        TODO: We're also going to need a out_spec (to handle pytree output types).
        """
        return super().__call__(func_spec, in_spec, *args)


flat_apply = FlatApply()


@flat_apply.py_impl(DispatchKey.CompositeExplicitAutograd)
def decomp(func_spec, in_spec, *args):
    func = pytree.tree_unflatten([], func_spec).func
    args, kwargs = unflatten(args, in_spec)
    out = func(*args, **kwargs)
    return out


@flat_apply.py_impl(ProxyTorchDispatchMode)
def _(proxy_mode, func_spec, in_spec, *args):
    qualname = proxy_mode.tracer.get_fresh_qualname("func_spec")
    setattr(proxy_mode.tracer.root, qualname, func_spec)
    func_proxy = proxy_mode.tracer.create_proxy("get_attr", qualname, (), {})

    qualname = proxy_mode.tracer.get_fresh_qualname("in_spec")
    setattr(proxy_mode.tracer.root, qualname, in_spec)
    in_spec_proxy = proxy_mode.tracer.create_proxy("get_attr", qualname, (), {})

    node_args = (func_proxy, in_spec_proxy, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", flat_apply, proxy_args, {}
    )
    out = decomp(func_spec, in_spec, *args)
    return track_tensor_tree(
        out, out_proxy, constant=None, tracer=proxy_mode.tracer  # type: ignore[arg-type]
    )


flat_apply.fallthrough(DispatchKey.AutogradCPU)
