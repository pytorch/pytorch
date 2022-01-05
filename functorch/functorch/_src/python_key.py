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

aten = torch.ops.aten

CURRENT_DECOMPOSITION_TABLE = {}
USE_META = False


@contextmanager
def pythonkey_decompose(decomposition_table):
    global CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = {}


@contextmanager
def pythonkey_meta():
    global USE_META
    USE_META = True
    try:
        yield USE_META
    finally:
        USE_META = False


def get_output_device(devices):
    if len(devices) == 1:
        return devices[0]
    else:
        for device in devices:
            if device.type == 'cuda':
                return device
        raise RuntimeError("Couldn't infer output device from input device")


class PythonTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem', 'proxy']

    @staticmethod
    def __new__(cls, elem, proxy, device=None):
        # The wrapping tensor (PythonTensor) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...

        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            dtype=elem.dtype, layout=elem.layout, requires_grad=elem.requires_grad,
            device=(elem.device if device is None else device),
        )

        # ...the real tensor is held as an element on the tensor.
        if USE_META:
            r.elem = elem.to('meta')
        else:
            r.elem = elem
        r.proxy = proxy
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(r)
        return r

    def __repr__(self):
        return f"PythonTensor({self.elem})"

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func in CURRENT_DECOMPOSITION_TABLE:
            return CURRENT_DECOMPOSITION_TABLE[func](*args, **kwargs)

        def unwrap_proxy(e):
            return e.proxy if isinstance(e, PythonTensor) else e

        def unwrap_tensor(e):
            return e.elem if isinstance(e, PythonTensor) else e

        input_devices = list(set([i.device for i in pytree.tree_flatten(args)[0] +
                                  pytree.tree_flatten(kwargs)[0] if isinstance(i, PythonTensor)]))
        output_device = get_output_device(input_devices)

        proxy_args = pytree.tree_map(unwrap_proxy, args)
        proxy_kwargs = pytree.tree_map(unwrap_proxy, kwargs)
        proxy_out = func(*proxy_args, **proxy_kwargs)

        # Kind of a hacky way to test if an op is in-place or not
        if func.__name__[-1] == "_" and func.__name__[0] != "_":
            args[0].proxy = proxy_out
        args = pytree.tree_map(unwrap_tensor, args)
        kwargs = pytree.tree_map(unwrap_tensor, kwargs)

        try:
            real_out = func(*args, **kwargs)
        except NotImplementedError:
            # Hardcoding in running in cuda if meta-tracing fails for now.
            args = pytree.tree_map(lambda x: torch.ones_like(x, device=output_device)
                                   if isinstance(x, torch.Tensor) else x, args)
            kwargs = pytree.tree_map(lambda x: torch.ones_like(x, device=output_device)
                                     if isinstance(x, torch.Tensor) else x, kwargs)
            real_out = func(*args, **kwargs)

        def wrap_with_proxy(e, proxy):
            # Some ops (like native_batch_norm_backward) return undefined tensors that get
            # converted into None in python.
            # As the function signature expects tensors, if we directly return these None
            # tensors back to C++, we'll error.
            if e is None:
                e = torch.empty(())
            if type(e) == torch.Tensor:
                return PythonTensor(e, proxy, output_device)
            else:
                return e
        if isinstance(real_out, tuple):
            return tuple([wrap_with_proxy(e, proxy_out[idx]) for idx, e in enumerate(real_out)])
        elif isinstance(real_out, list):
            return list([wrap_with_proxy(e, proxy_out[idx]) for idx, e in enumerate(real_out)])
        elif isinstance(real_out, torch.Tensor):
            return wrap_with_proxy(real_out, proxy_out)
        else:
            return real_out


class PythonKeyTracer(Tracer):
    def __init__(self):
        super().__init__()

    def call_module(
        self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        return forward(*args, **kwargs)

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if isinstance(attr_val, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        proxy = self.create_proxy('get_attr', n, (), {})
                        parameter_proxy_cache[n] = PythonTensor(attr_val, proxy)
                    return parameter_proxy_cache[n]
            return attr_val
        return attr_val

    # We need to do this so that parameters entering the `make_fx` context have
    # a reference to them (and also have requires_grad set on them correctly
    # I'm not actually sure if this is the right thing to do ...
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


def pythonkey_trace(
    root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None
) -> GraphModule:
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


def make_fx(f, decomposition_table={}):
    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda x: fx.PH, args)
        with pythonkey_decompose(decomposition_table):
            t = pythonkey_trace(wrap_key(f, args), concrete_args=tuple(phs))
        return t

    return wrapped
