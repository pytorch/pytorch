# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Any, Dict, Optional, Tuple, Callable, Union
import torch
from torch._C import _disabled_torch_function_impl
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager

from torch.utils._python_dispatch import push_torch_dispatch_mode, TorchDispatchMode

__all__ = ["ProxyTensor", "PythonKeyTracer", "dispatch_trace", "make_fx"]
aten = torch.ops.aten

CURRENT_DECOMPOSITION_TABLE: Dict[torch._ops.OpOverload, Callable] = {}


@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


@contextmanager
def decompose(decomposition_table):
    global CURRENT_DECOMPOSITION_TABLE
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table


def wrap_output(real_out, proxy_out):
    def wrap_with_proxy(e, proxy):
        if type(e) == torch.Tensor:
            with no_dispatch():
                return ProxyTensor(e, proxy)
        else:
            return e

    # Unfortunately, tree_map cannot directly be used here. As the resulting
    # object may be a proxy that represents a tuple, we may need to
    # explicitly unwrap the proxy by simulating the flattening operations.
    if isinstance(real_out, tuple):
        return tuple(wrap_with_proxy(e, proxy_out[idx]) for idx, e in enumerate(real_out))
    elif isinstance(real_out, list):
        return list([wrap_with_proxy(e, proxy_out[idx]) for idx, e in enumerate(real_out)])
    elif isinstance(real_out, torch.Tensor):
        return wrap_with_proxy(real_out, proxy_out)
    else:
        return real_out


def proxy_call(func_overload, args, kwargs=None):
    func = func_overload.overloadpacket
    if func_overload in CURRENT_DECOMPOSITION_TABLE:
        return CURRENT_DECOMPOSITION_TABLE[func_overload](*args, **kwargs)
    if func_overload == aten._local_scalar_dense.default:
        raise RuntimeError("It appears that you're trying to get value out of a tracing tensor - erroring out! "
                           "It's likely that this is caused by data-dependent control flow or similar.")

    def unwrap_proxy(e):
        return e.proxy if isinstance(e, ProxyTensor) else e

    proxy_args = pytree.tree_map(unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(unwrap_proxy, kwargs)

    proxy_out = func(*proxy_args, **proxy_kwargs)

    # Kind of a hacky way to test if an op is in-place or not
    if func.__name__[-1] == "_" and func.__name__[0] != "_":
        args[0].proxy = proxy_out
        proxy_out.node.meta['tensor_meta'] = _extract_tensor_metadata(args[0])

    with no_dispatch():
        real_out = func_overload(*args, **kwargs)

    return wrap_output(real_out, proxy_out)

class ProxyTensor(torch.Tensor):
    proxy: fx.Proxy

    @staticmethod
    def __new__(cls, elem, proxy):
        # Hack to deal with super().__new__ not working for sparse tensors
        if elem.is_sparse:
            proxy.node.meta['tensor_meta'] = {}
            r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        else:
            r = super().__new__(cls, elem)  # type: ignore[call-arg]
            proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(r)
        r.proxy = proxy  # type: ignore[attr-defined]

        return r

    def __repr__(self):
        with no_dispatch():
            return f"ProxyTensor({self.as_subclass(torch.Tensor)}, proxy={self.proxy})"  # type: ignore[arg-type]

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        return proxy_call(func_overload, args, kwargs)


class PythonKeyTracer(Tracer):
    def __init__(self):
        super().__init__()

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
            self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        return forward(*args, **kwargs)

    def create_arg(self, a: Any):
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            qualname: Optional[str] = None

            if not qualname:
                i = 0
                while True:
                    qualname = f'_param_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})
        return super().create_arg(a)


def dispatch_trace(
        root: Union[torch.nn.Module, Callable],
        concrete_args: Optional[Tuple[Any, ...]] = None,
        trace_factory_functions: bool = False,
) -> GraphModule:
    tracer = PythonKeyTracer()
    if trace_factory_functions:
        with push_torch_dispatch_mode(functools.partial(ProxyTorchDispatchMode, tracer)):
            graph = tracer.trace(root, concrete_args)
    else:
        graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


def wrap_key(f, inps):
    flat_inps, _ = pytree.tree_flatten(inps)

    @functools.wraps(f)
    def wrapped(*args):
        flat_args, args_spec = pytree.tree_flatten(args)
        assert (len(flat_args) == len(flat_inps))
        for idx, arg in enumerate(flat_args):
            if isinstance(flat_inps[idx], torch.Tensor):
                with no_dispatch():
                    flat_args[idx] = ProxyTensor(flat_inps[idx], arg)
            else:
                flat_args[idx] = flat_inps[idx]

        tree_args = pytree.tree_unflatten(flat_args, args_spec)
        out = f(*tree_args)
        flat_outs, out_spec = pytree.tree_flatten(out)
        for idx in range(len(flat_outs)):
            if isinstance(flat_outs[idx], torch.Tensor) and isinstance(flat_outs[idx], ProxyTensor):
                flat_outs[idx] = flat_outs[idx].proxy
        return pytree.tree_unflatten(flat_outs, out_spec)

    return wrapped


class ProxyTorchDispatchMode(TorchDispatchMode):
    def __init__(self, tracer):
        self.tracer = tracer

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        func = func_overload.overloadpacket
        if any(tuple(isinstance(arg, ProxyTensor) for arg in args)):
            return proxy_call(func_overload, args, kwargs)
        else:
            proxy_out = self.tracer.create_proxy('call_function', func, args, kwargs,
                                                 name=self.tracer.graph._target_to_str(func.__name__))

            with no_dispatch():
                real_out = func_overload(*args, **kwargs)

            return wrap_output(real_out, proxy_out)


def make_fx(f, decomposition_table=None, trace_factory_functions=False):
    if decomposition_table is None:
        decomposition_table = {}

    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda x: fx.PH, args)  # type: ignore[attr-defined]
        with decompose(decomposition_table):
            t = dispatch_trace(wrap_key(f, args), concrete_args=tuple(phs),
                               trace_factory_functions=trace_factory_functions)
        return t

    return wrapped
