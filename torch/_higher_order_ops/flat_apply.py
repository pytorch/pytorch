from weakref import proxy
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
    # Separate the graphable vs non-graphable
    sanitized_flat_args = []
    non_graphable = []
    for idx, arg in enumerate(flat_args):
        if is_graphable(arg):
            sanitized_flat_args.append(arg)
        else:
            non_graphable.append((idx, arg))

    return sanitized_flat_args, spec, non_graphable


def unflatten(graphable_flat_args, spec, non_graphable):
    numel = len(graphable_flat_args) + len(non_graphable)

    graphable_iter = list(graphable_flat_args)
    non_graphable_iter = list(non_graphable)

    flat_args = []
    for idx in range(numel):
        if len(non_graphable) > 0 and non_graphable[0][0] == idx:
            flat_args.append(non_graphable_iter[0][1])
            non_graphable_iter.pop(0)  # TODO: slow
        else:
            flat_args.append(graphable_iter[0])
            graphable_iter.pop(0)  # TODO: slow

    stuff = pytree.tree_unflatten(flat_args, spec)
    return stuff


def plop_in_graph(f):
    def inner(*args, **kwargs):
        flat_args, spec, non_graphable = flatten((args, kwargs))
        f_key = ConstantHolder(f)
        in_spec = ConstantHolder((spec, non_graphable))
        return flat_apply(f_key, in_spec, *flat_args)

    return inner


class ConstantHolder:
    def __init__(self, value):
        self.value = value


class FlatApply(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_apply")

    def __call__(self, func_holder, in_spec_holder, *args):
        """
        The semantics of flat_apply(func_holder, in_spec_holder, *args) is the following:

        >>> func = func_holder.value
        >>> in_spec = in_spec_holder.vallue
        >>> args, kwargs = unflatten(args, in_spec)
        >>> output = func(*args, **kwargs)
        >>> return output

        TODO: We're also going to need a out_spec (to handle pytree output types).
        """
        return super().__call__(func_holder, in_spec_holder, *args)


flat_apply = FlatApply()


@flat_apply.py_impl(DispatchKey.CompositeExplicitAutograd)
def decomp(func_holder, in_spec_holder, *args):
    func = func_holder.value
    in_spec = in_spec_holder.value
    args, kwargs = unflatten(args, *in_spec)
    out = func(*args, **kwargs)
    return out


@flat_apply.py_impl(ProxyTorchDispatchMode)
def _(proxy_mode, func_holder, in_spec_holder, *args):

    qualname = proxy_mode.tracer.get_fresh_qualname("func_holder")
    setattr(proxy_mode.tracer.root, qualname, func_holder)
    func_proxy = proxy_mode.tracer.create_proxy("get_attr", qualname, (), {})

    qualname = proxy_mode.tracer.get_fresh_qualname("in_spec_holder")
    setattr(proxy_mode.tracer.root, qualname, in_spec_holder)
    in_spec_proxy = proxy_mode.tracer.create_proxy("get_attr", qualname, (), {})

    node_args = (func_proxy, in_spec_proxy, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", flat_apply, proxy_args, {}
    )
    out = decomp(func_holder, in_spec_holder, *args)
    return track_tensor_tree(
        out, out_proxy, constant=None, tracer=proxy_mode.tracer  # type: ignore[arg-type]
    )


flat_apply.fallthrough(DispatchKey.AutogradCPU)
