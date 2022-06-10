# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Any, Dict, Optional, Tuple, Callable, Union
import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.fake_symbolic_tensor import PySymInt
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
import torch.fx as fx
from torch.utils._mode_utils import no_dispatch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager
from .fake_symbolic_tensor import FakeSymbolicTensor, create_contiguous, ShapeEnv

from torch.utils._python_dispatch import push_torch_dispatch_mode, TorchDispatchMode

__all__ = ["ProxyTensor", "PythonKeyTracer", "dispatch_trace", "make_fx", "enable_strict"]
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

def wrap_output(real_out, proxy_out):
    def wrap_with_proxy(e, proxy):
        if isinstance(e, torch.Tensor):
            with no_dispatch():
                return ProxyTensor(proxy_out, e.shape, create_contiguous(e.shape), e.dtype, e.layout, e.requires_grad, e.device)
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

from torch.overrides import enable_reentrant_dispatch


class ProxyTensor(FakeSymbolicTensor):
    proxy: fx.Proxy

    @staticmethod
    def __new__(cls, proxy, sym_shape, sym_strides, dtype, layout, requires_grad, device):
        # Hack to deal with super().__new__ not working for sparse tensors
        def create_proxy_symint(sym_int, new_proxy):
            return torch._C.SymbolicIntNode.new_symint(ProxySymInt(sym_int, new_proxy))

        r = super().__new__(cls, [create_proxy_symint(sym_shape[i], proxy.size(i)) for i in range(len(sym_shape))], sym_strides, dtype, layout, requires_grad, device)
        r.proxy = proxy
        # if elem.is_sparse:
        #     proxy.node.meta['tensor_meta'] = {}
        # else:
        #     proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(r)

        return r

    def __deepcopy__(self, memo):
        return self.clone()

    def __repr__(self):
        with no_dispatch():
            return f"ProxyTensor({self.as_subclass(torch.Tensor)}, proxy={self.proxy})"  # type: ignore[arg-type]

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        # print(func_overload)
        func = func_overload.overloadpacket
        if func_overload in CURRENT_DECOMPOSITION_TABLE:
            return CURRENT_DECOMPOSITION_TABLE[func_overload](*args, **kwargs)
        if func_overload == aten._local_scalar_dense.default:
            raise RuntimeError("It appears that you're trying to get value out of a tracing tensor - erroring out! "
                            "It's likely that this is caused by data-dependent control flow or similar."
                            "Try torch.fx.experimental.proxy_tensor.enable_strict(False) to disable this check")

        def unwrap_proxy(e):
            return e.proxy if isinstance(e, ProxyTensor) else e

        def unwrap_proxyints(shape):
            return [e.get_pyobj().sym_int if isinstance(e.get_pyobj(), ProxySymInt) else e for e in shape]

        def unwrap_fake(e):
            if isinstance(e, ProxyTensor):
                return FakeSymbolicTensor(unwrap_proxyints(e.shape), None, e.dtype, e.layout, e.requires_grad, e.device)
            return e

        proxy_args = pytree.tree_map(unwrap_proxy, args)
        proxy_kwargs = pytree.tree_map(unwrap_proxy, kwargs)

        proxy_out = func(*proxy_args, **proxy_kwargs)

        # Kind of a hacky way to test if an op is in-place or not
        if func.__name__[-1] == "_" and func.__name__[0] != "_":
            args[0].proxy = proxy_out
            proxy_out.node.meta['tensor_meta'] = _extract_tensor_metadata(args[0])

        real_out = super().__torch_dispatch__(func_overload, types, pytree.tree_map(unwrap_fake, args), pytree.tree_map(unwrap_fake, kwargs))

        return wrap_output(real_out, proxy_out)


class ProxySymInt(object):
    def __init__(self, sym_int, proxy):
        assert isinstance(sym_int, torch._C.SymbolicIntNode) or isinstance(sym_int, int)
        self.sym_int = sym_int
        self.proxy = proxy

    def wrap(self, num):
        return ProxySymInt(num, num)

    def __str__(self):
        return f"ProxySymInt({self.sym_int})"

    def __int__(self):
        return int(self.sym_int)

    def __bool__(self):
        return bool(self.sym_int)

magic_methods = [
    'add',
    # 'radd',
    'sub',
    'mul',
    # 'div',
    'mod',
    'eq',
    'gt',
    'lt',
]

import operator

for method in magic_methods:
    method_name = f'{method}'
    op = getattr(operator, method_name)
    def create_magic_impl(op):
        def magic_impl(self, other):
            def unwrap_proxy(x): return x.proxy if isinstance(x, ProxySymInt) else x
            out_proxy = op(unwrap_proxy(self), unwrap_proxy(other))
            def unwrap_proxyint(x): return x.sym_int if isinstance(x, ProxySymInt) else x
            out_sym_int = op(unwrap_proxyint(self), unwrap_proxyint(other))
            return ProxySymInt(out_sym_int, out_proxy)
        return magic_impl

    # this should be wrapped transparently into torch._C.SymbolicIntNode
    setattr(ProxySymInt, method_name, create_magic_impl(op))

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
        elif isinstance(a, torch._C.SymbolicIntNode):
            py_symint = a.get_pyobj()
            assert isinstance(py_symint, ProxySymInt)
            return py_symint.proxy.node
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


def create_proxy_tensor_symbolic(proxy, arg, shape_env):
    sym_shapes = tuple([shape_env.create_symint(f"{proxy.node.name}_{idx}", val) for idx, val in enumerate(arg.size())])
    sym_strides = tuple([shape_env.create_symint(f"{proxy.node.name}_{idx}_stride", val) for idx, val in enumerate(arg.stride())])
    return ProxyTensor(proxy, sym_shapes, sym_strides, arg.dtype, arg.layout, arg.requires_grad, arg.device)


def wrap_key(f, inps, create_proxy_tensor):
    flat_inps, _ = pytree.tree_flatten(inps)

    @functools.wraps(f)
    def wrapped(*args):
        flat_args, args_spec = pytree.tree_flatten(args)
        assert (len(flat_args) == len(flat_inps))
        for idx, arg in enumerate(flat_args):
            if isinstance(flat_inps[idx], torch.Tensor):
                with no_dispatch():
                    flat_args[idx] = create_proxy_tensor(flat_args[idx], flat_inps[idx])
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

from functools import partial


def make_fx(f, decomposition_table=None, trace_factory_functions=False):
    if decomposition_table is None:
        decomposition_table = {}
    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda x: fx.PH, args)  # type: ignore[attr-defined]
        shape_env = ShapeEnv()
        with decompose(decomposition_table):
            t = dispatch_trace(wrap_key(f, args, partial(create_proxy_tensor_symbolic, shape_env = shape_env)), concrete_args=tuple(phs),
                               trace_factory_functions=trace_factory_functions)
            t.shape_env = shape_env
        return t

    return wrapped
