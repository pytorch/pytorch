# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, List, Callable, Union
import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.node import map_aggregate
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
import torch.fx as fx
import torch.fx._pytree as fx_pytree
from .nnc_compile import nnc_compile

class PythonTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem', 'proxy']

    @staticmethod
    def __new__(cls, elem, proxy):
        # The wrapping tensor (PythonTensor) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        meta = elem.new_empty((0,))
        meta.set_(meta.storage(), 0, elem.size(), elem.stride())
        r = torch.Tensor._make_subclass(cls, meta, elem.requires_grad)

        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        r.proxy = proxy
        return r

    def __repr__(self):
        return f"PythonTensor({self.elem})"

    __torch_function__ = _disabled_torch_function_impl
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap_proxy(e):
            return e.proxy if isinstance(e, PythonTensor) else e

        def unwrap_tensor(e):
            return e.elem if isinstance(e, PythonTensor) else e
        aten_func = getattr(torch.ops.aten, func.__name__)
        proxy_args = pytree.tree_map(unwrap_proxy, args)
        proxy_out = aten_func(*proxy_args)
        real_out = aten_func(*pytree.tree_map(unwrap_tensor, args))

        def wrap_with_proxy(e, idx):
            return PythonTensor(e, proxy_out[idx]) if type(e) == torch.Tensor else e

        if isinstance(real_out, tuple):
            return tuple([wrap_with_proxy(e, idx) for idx, e in enumerate(real_out)])
        elif isinstance(real_out, list):
            return list([wrap_with_proxy(e, idx) for idx, e in enumerate(real_out)])
        else:
            return PythonTensor(real_out, proxy_out) if type(real_out) ==  torch.Tensor else real_out

class PythonKeyTracer(Tracer):
    def __init__(self):
        super().__init__()


    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        return forward(*args, **kwargs)

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if isinstance(attr_val, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        proxy = self.create_proxy('get_attr', n, (), {})
                        parameter_proxy_cache[n] = PythonTensor(attr_val, proxy)
                    return parameter_proxy_cache[n]
            return attr_val.data
        return attr_val

def pythonkey_trace(root : Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:
    tracer = PythonKeyTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)

def wrap_key(f, inps):
    flat_inps, inp_spec = pytree.tree_flatten(inps)
    @functools.wraps(f)
    def wrapped(*args):
        flat_args, args_spec = pytree.tree_flatten(args)
        assert(len(flat_args) == len(flat_inps))
        for idx, arg in enumerate(flat_args):
            if isinstance(flat_inps[idx], torch.Tensor):
                flat_args[idx] = PythonTensor(flat_inps[idx], arg)
            else:
                flat_args[idx] = flat_inps[idx]

        tree_args = pytree.tree_unflatten(flat_args, args_spec)
        out = f(*tree_args)
        flat_outs, out_spec = pytree.tree_flatten(out)
        for idx in range(len(flat_outs)):
            if isinstance(flat_outs[idx], torch.Tensor) and isinstance(flat_outs[idx], PythonTensor):
                flat_outs[idx] = flat_outs[idx].proxy
        return pytree.tree_unflatten(flat_outs, out_spec)

    return wrapped

def make_fx(f):
    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda x: fx.PH, args)
        t = pythonkey_trace(wrap_key(f, args), concrete_args=tuple(phs))
        return t

    return wrapped


def nnc_jit(f):
    cached = None
    @functools.wraps(f)
    def compiled(*args):
        nonlocal cached
        if cached is not None:
            return cached(*args)
        fx_model = make_fx(f)(*args)
        fx_model.graph.lint()
        compiled_f = nnc_compile(fx_model, args)
        cached = compiled_f
        return cached(*args)
    return compiled

def make_nnc(f):
    @functools.wraps(f)
    def wrapped(*args):
        fx_model = make_fx(f)(*args)
        fx_model.graph.lint()
        compiled_f = nnc_compile(fx_model, args, get_loopnest=True)
        return compiled_f

    return wrapped
