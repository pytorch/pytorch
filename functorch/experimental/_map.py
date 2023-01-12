from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch._ops import PyOperator
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


map = PyOperator("map")


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


@map.py_impl(DispatchKey.CPU)
def map_cpu(f, xs, *args):
    mode = _get_current_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU key"
    return torch.stack([f(x, *args) for x in xs])


@map.py_impl(DispatchKey.AutogradCPU)
def map_autograd(f, xs, *args):
    # TODO: support autograd
    flat_operands, _ = tree_flatten([f, xs, args])
    assert all([not f.requires_grad for f in flat_operands
                if isinstance(f, torch.Tensor)])

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

# We cannot directly call fallthrough here due to issue #89037.
@map.py_impl(DispatchKey.PythonDispatcher)
def map_python_dispatcher(*args):
    _ = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.PythonDispatcher))
    return map(*args)


# TODO(voz) Make this automatic for keys, this is very ugly atm
map.fallthrough(DispatchKey.PythonTLSSnapshot)
map.fallthrough(DispatchKey.ADInplaceOrView)
map.fallthrough(DispatchKey.BackendSelect)
