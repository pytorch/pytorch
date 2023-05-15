from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
    unwrap_proxy,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch.utils._pytree import tree_flatten
from ._cond import _has_potential_branch_input_alias, _has_potential_branch_input_mutation, UnsupportedAliasMutationException


map = HigherOrderOperator("map")


def trace_map(proxy_mode, func_overload, f, xs, *args):
    if not isinstance(xs, torch.Tensor):
        raise ValueError("map() must loop over a tensor")
    if len(xs.shape) == 0 or xs.shape[0] == 0:
        raise ValueError("map() cannot be traced with scalar tensors or zero dimension tensors")
    if not all(isinstance(o, torch.Tensor) for o in args):
        raise ValueError("map() operands must be a list of tensors or modules")

    with disable_proxy_modes_tracing():
        body_graph = make_fx(f)(xs[0], *args)

    next_name = None
    i = 0
    while not next_name:
        candidate = f"body_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    proxy_mode.tracer.root.register_module(next_name, body_graph)
    node_args = (body_graph, xs, *args)
    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
                                               name="map")
    outs = [body_graph(x, *args) for x in xs]
    # Implementation notes: we need to use new_empty() + copy_() here instead of stack() directly
    # because stack([...]) takes a fixed size list which will specialize dynamic shape here.
    # Meanwhile we want to preserve the looped over dimension as symbolic shape, such that:
    # ys: Tensor[s0, ...] = map(xs: Tensor[s0, ...], *args)
    out = outs[0].new_empty([xs.shape[0], *outs[0].shape])
    out.copy_(torch.stack(outs))
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@map.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_cpu(f, xs, *args):
    mode = _get_current_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU/CUDA key"
    return torch.stack([f(x, *args) for x in xs])


@map.py_impl(DispatchKey.Autograd)
def map_autograd(f, xs, *args):
    # TODO: support autograd
    flat_operands, _ = tree_flatten([f, xs, args])
    assert all(not f.requires_grad for f in flat_operands
               if isinstance(f, torch.Tensor))

    _ = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.AutogradCPU))
    return map(f, xs, *args)


@map.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(f, xs, *args):
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        res = trace_map(mode, map, f, xs, *args)
    return res


@map.py_impl(FakeTensorMode)
def map_fake_tensor_mode(f, xs, *args):
    outs = [f(x, *args) for x in xs]
    return outs[0].new_empty([xs.shape[0], *outs[0].shape])


@map.py_impl(DispatchKey.Functionalize)
def map_func(f, xs, *args):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(args, reapply_views=reapply_views)
    mode = 'mutations_and_views' if reapply_views else 'mutations'

    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize))
    try:
        functional_map_fn = functionalize(f, remove=mode)
        inputs = (unwrapped_xs,) + unwrapped_args

        if _has_potential_branch_input_mutation(f, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(f, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map(functional_map_fn, unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=0)
    finally:
        del guard

@map.py_impl(torch._C._functorch.TransformType.Functionalize)
def map_functionalize(interpreter, f, xs, *args):
    """
    Functionalization implementation for torch.map. Currently:
      1. We don't allow any input mutation inside the map function
      2. Our check for above condition is not exhaustive
    """
    reapply_views = interpreter.functionalize_add_back_views()
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(args, reapply_views=reapply_views)

    functional_map_fn = functionalize(f, remove=mode)

    with interpreter.lower():
        inputs = (unwrapped_xs,) + unwrapped_args
        if _has_potential_branch_input_mutation(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map(functional_map_fn, unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=interpreter.level())

# TODO(voz) Make this automatic for keys, this is very ugly atm
map.fallthrough(DispatchKey.PythonDispatcher)
map.fallthrough(DispatchKey.PythonTLSSnapshot)
map.fallthrough(DispatchKey.ADInplaceOrView)
map.fallthrough(DispatchKey.BackendSelect)
map.fallthrough(DispatchKey.AutocastCPU)
