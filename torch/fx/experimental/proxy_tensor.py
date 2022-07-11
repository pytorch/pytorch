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
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.fx as fx
from torch.utils._mode_utils import no_dispatch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext

from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["ProxyTensor", "PythonKeyTracer", "dispatch_trace", "make_fx", "enable_strict", "DecompositionInterpreter"]
aten = torch.ops.aten

CURRENT_DECOMPOSITION_TABLE: Dict[torch._ops.OpOverload, Callable] = {}


@contextmanager
def decompose(decomposition_table):
    global CURRENT_DECOMPOSITION_TABLE
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table

# Checks whether we try to convert the tensor into a scalar
IS_STRICT = True
def enable_strict(val):
    global IS_STRICT
    IS_STRICT = val

def wrap_output(inner_res, proxy_res):
    def wrap_with_proxy(e, proxy):
        if isinstance(e, torch.Tensor):
            with no_dispatch():
                return ProxyTensor(e, proxy)
        else:
            return e

    # Unfortunately, tree_map cannot directly be used here. As the resulting
    # object may be a proxy that represents a tuple, we may need to
    # explicitly unwrap the proxy by simulating the flattening operations.
    if isinstance(inner_res, tuple):
        return tuple(wrap_with_proxy(e, proxy_res[idx]) for idx, e in enumerate(inner_res))
    elif isinstance(inner_res, list):
        return list([wrap_with_proxy(e, proxy_res[idx]) for idx, e in enumerate(inner_res)])
    elif isinstance(inner_res, torch.Tensor):
        return wrap_with_proxy(inner_res, proxy_res)
    else:
        return inner_res


def proxy_call(func_overload, args, kwargs=None):
    if kwargs is None:
        kwargs = {}
    func = func_overload.overloadpacket
    if func_overload in CURRENT_DECOMPOSITION_TABLE:
        return CURRENT_DECOMPOSITION_TABLE[func_overload](*args, **kwargs)
    if func_overload == aten._local_scalar_dense.default:
        raise RuntimeError("It appears that you're trying to get value out of a tracing tensor - erroring out! "
                           "It's likely that this is caused by data-dependent control flow or similar."
                           "Try torch.fx.experimental.proxy_tensor.enable_strict(False) to disable this check")

    def unwrap_proxy(e):
        return e.proxy if isinstance(e, ProxyTensor) else e

    def unwrap_elem(e):
        if isinstance(e, ProxyTensor):
            return e.elem
        return e

    proxy_args = pytree.tree_map(unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(unwrap_proxy, kwargs)

    proxy_res = func_overload(*proxy_args, **proxy_kwargs)

    # Kind of a hacky way to test if an op is in-place or not
    if func.__name__[-1] == "_" and func.__name__[0] != "_":
        args[0].proxy = proxy_res
        proxy_res.node.meta['tensor_meta'] = _extract_tensor_metadata(args[0])

    inner_res = func_overload(*pytree.tree_map(unwrap_elem, args), **pytree.tree_map(unwrap_elem, kwargs))
    # Needed to sync up metadata for in-place operators that modify metadata
    if torch.Tag.inplace_view in func_overload.tags:  # type: ignore[attr-defined]
        with no_dispatch():
            func_overload(*args, **kwargs)

    # TODO(chilli): Enable this after it's been refactored to work with wrapper tensor subclasses in general
    # pytree.tree_map(lambda x: check_metadata_consistency(x, ProxyTensor), (inner_res, args, kwargs))
    return wrap_output(inner_res, proxy_res)


class ProxyTensor(torch.Tensor):
    proxy: fx.Proxy
    elem: torch.Tensor


    @staticmethod
    def __new__(cls, elem, proxy, *, requires_grad=None):
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            elem.shape, dtype=elem.dtype, layout=elem.layout, device=elem.device,
            requires_grad=requires_grad if requires_grad is not None else False, strides=elem.stride(),
            storage_offset=elem.storage_offset()
        )
        return r

    def __init__(self, elem, proxy, *, requires_grad=None):
        if elem.is_sparse:
            proxy.node.meta['tensor_meta'] = {}
        else:
            proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(self)
        self.elem = elem
        self.proxy = proxy

    def __deepcopy__(self, memo):
        return self.clone()

    def __repr__(self):
        with no_dispatch():
            return f"ProxyTensor({self.elem}, proxy={self.proxy})"

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
        tracer: Tracer,
        concrete_args: Optional[Tuple[Any, ...]] = None,
) -> GraphModule:
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
                    flat_args[idx] = ProxyTensor(
                        flat_inps[idx],
                        arg,
                        requires_grad=(flat_inps[idx].is_leaf and flat_inps[idx].requires_grad)
                    )
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
        if any(tuple(isinstance(arg, ProxyTensor) for arg in pytree.tree_flatten(args)[0])):
            return proxy_call(func_overload, args, kwargs)
        else:
            proxy_res = self.tracer.create_proxy('call_function', func_overload, args, kwargs,
                                                 name=self.tracer.graph._target_to_str(func.__name__))

            inner_res = func_overload(*args, **kwargs)

            return wrap_output(inner_res, proxy_res)


class DecompositionInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule, new_graph: torch.fx.Graph, decomposition_table=None, **kwargs):
        super().__init__(module, **kwargs)
        self.new_graph = new_graph
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)
        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        # TODO handle case where the first character of target is '*'
        return ProxyTensor(out, torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer))

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        return ProxyTensor(out, torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer))

    # call_function, call_method, call_module get traced automatically by the ProxyTensors.

    def output(self, target, args, kwargs):
        out = super().output(target, args, kwargs)

        def unwrap(e):
            return e.proxy.node if isinstance(e, ProxyTensor) else e
        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args, **kwargs):
        with decompose(self.decomposition_table):
            return super().run(*args, **kwargs)

def make_fx(f, decomposition_table=None, trace_factory_functions=True, use_fake=False):
    if decomposition_table is None:
        decomposition_table = {}

    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda _: fx.PH, args)  # type: ignore[attr-defined]
        fx_tracer = PythonKeyTracer()
        fake_tensor_mode = FakeTensorMode() if use_fake else nullcontext()
        proxy_mode = ProxyTorchDispatchMode(fx_tracer) if trace_factory_functions else nullcontext()

        def wrap_fake(x):
            if isinstance(x, torch.Tensor):
                return fake_tensor_mode.from_tensor(x)  # type: ignore[attr-defined]

            return x

        if use_fake:  # type: ignore[attr-defined]
            args = pytree.tree_map(wrap_fake, args)

        with decompose(decomposition_table), fake_tensor_mode, proxy_mode:  # type: ignore[attr-defined]
            t = dispatch_trace(wrap_key(f, args), tracer=fx_tracer, concrete_args=tuple(phs))
        return t

    return wrapped
