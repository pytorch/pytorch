from typing import Callable, Sequence, Any, Dict
import functools


import torch
import torch.overrides
from torch.fx.experimental.proxy_tensor import ProxyTensor, make_fx

from torch._prims_common import torch_function_passthrough

import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special

import torch._prims


@functools.lru_cache(None)
def torch_to_refs_map():
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
    modules = [
        (torch, torch._refs),
        (torch.nn, torch._refs.nn),
        (torch.nn.functional, torch._refs.nn.functional),
        (torch.special, torch._refs.special),
        (torch.fft, torch._refs.fft),
    ]
    r: Dict[Any, Any] = {
        torch.Tensor.__invert__: torch._refs.bitwise_not,
        torch.Tensor.__xor__: torch._refs.bitwise_xor,
        torch.Tensor.__and__: torch._refs.bitwise_and,
        torch.Tensor.__or__: torch._refs.bitwise_or,
        torch.Tensor.__eq__: torch._refs.eq,
        # TODO: Should these methods be mapped some other way?
        torch.Tensor.copy_: torch._prims.copy_to,
        torch.Tensor.resize: torch._prims.resize,
    }
    for mod_torch, mod_refs in modules:
        for s in mod_refs.__all__:  # type: ignore[attr-defined]
            r[mod_torch.__dict__.get(s)] = mod_refs.__dict__.get(s)

    # Support remapping torch.Tensor.foo to _refs.foo
    for s in dir(torch.Tensor):
        if s in torch._refs.__all__:
            r[getattr(torch.Tensor, s)] = torch._refs.__dict__.get(s)
    return r


@functools.lru_cache(None)
def all_prims():
    """
    Set of all prim functions, e.g., torch._prims.add in all_prims()
    """
    return {torch._prims.__dict__.get(s) for s in torch._prims.__all__}


class TorchRefsMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    use PrimTorch refs in torch._refs.  (Direct calls to _refs are unaffected.)

    >>> with TorchRefsMode.push():
    ...     torch.add(x, y)  # calls torch._refs.add(x, y)

    By default, this context manager will fall back on the torch.* if the
    ref does not exist; set strict=True to error if this occurs.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __torch_function__(
        self,
        orig_func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Dict = None,
    ):
        if kwargs is None:
            kwargs = {}
        # For primitive operations, run them as is without interception
        if orig_func in torch_function_passthrough or orig_func in all_prims():
            return orig_func(*args, **kwargs)
        mapping = torch_to_refs_map()
        func = mapping.get(orig_func, None)
        if func is not None:
            # torch calls inside func should be interpreted as refs calls
            with torch.overrides.enable_torch_function_mode(self, replace=self.inner):
                return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(
                f"no _refs support for {torch.overrides.resolve_name(orig_func)}"
            )
        return orig_func(*args, **kwargs)

def _is_node_supported(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and getattr(node.target, "impl_nvfuser", None) is not None
    )

def _find_proxy_tensor(*objects_to_search):
    # return first proxy tensor found or None
    proxy_tensors = (o for o in objects_to_search if isinstance(o, ProxyTensor))
    return next(proxy_tensors, None)

def _get_subgraph(func, args, kwargs):
    # make_fx doesn't support kwargs, so we need to do this flattening
    # and then unflatten the args before calling func
    nargs = len(args)
    flat_kwargs = list(kwargs.values())
    all_args = list(args) + flat_kwargs

    def wrapped(args):
        fn_args = args[:nargs]
        kwargs_keys = list(kwargs.keys())
        fn_kwargs = dict(zip(kwargs_keys, args[nargs:]))
        return func(*fn_args, **fn_kwargs)

    # extract outer tracer object
    outer_tracer = _find_proxy_tensor(*all_args).proxy.tracer

    # create a new tracer object
    graph = torch.fx.Graph()
    new_tracer = torch.fx.experimental.proxy_tensor.PythonKeyTracer()
    new_tracer.graph = graph

    try:
        for arg in all_args:
            if isinstance(arg, torch.fx.experimental.proxy_tensor.ProxyTensor):
                arg.proxy.tracer = new_tracer

        gm = make_fx(wrapped)(all_args)
    finally:
        for arg in all_args:
            if isinstance(arg, torch.fx.experimental.proxy_tensor.ProxyTensor):
                arg.proxy.tracer = outer_tracer

    return gm

class TorchRefsNvfuserMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    use PrimTorch refs in torch._refs.  (Direct calls to _refs are unaffected.)
    only if the reference function can be run with nvFuser executor.

    >>> with TorchRefsNvfuserMode.push():
    ...     torch.add(x, y)  # calls torch._refs.add(x, y)

    By default, this context manager will fall back on the torch.* if the
    ref does not exist; set strict=True to error if this occurs.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __torch_function__(
        self,
        orig_func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Dict = None,
    ):
        if kwargs is None:
            kwargs = {}
        # For primitive operations, run them as is without interception
        if orig_func in torch_function_passthrough or orig_func in all_prims():
            return orig_func(*args, **kwargs)
        mapping = torch_to_refs_map()
        func = mapping.get(orig_func, None)
        if func is not None:
            # torch calls inside func should be interpreted as refs calls
            # try and check the subgraph

            with torch.overrides.enable_torch_function_mode(self, replace=self.inner):
                gm = _get_subgraph(func, args, kwargs)

            print(f"__torch_function__: orig_func={torch.overrides.resolve_name(orig_func)}")
            gm.graph.print_tabular()

            call_function_nodes = filter(lambda n: n.op == "call_function", gm.graph.nodes)
            any_unsupported = any(
                not _is_node_supported(node) for node in call_function_nodes
            )
            print(f"any_unsupported={any_unsupported}")
            if any_unsupported:
                return orig_func(*args, **kwargs)
            else:
                with torch.overrides.enable_torch_function_mode(self, replace=self.inner):
                    return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(
                f"no _refs support for {torch.overrides.resolve_name(orig_func)}"
            )
        return orig_func(*args, **kwargs)
