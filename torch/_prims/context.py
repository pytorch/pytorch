from typing import Callable, Sequence, Any, Dict
import functools


import torch
import torch.overrides
from torch.fx.experimental.proxy_tensor import get_isolated_graphmodule

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
    If the ref exists we still would like to fall back on the torch.* sometimes,
    this behavior can be customized by passing a function to should_fallback_fn.
    """

    def __init__(self, strict=False, should_fallback_fn=lambda *_: True):
        self.strict = strict
        self.should_fallback_fn = should_fallback_fn

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
            # If the ref exists query whether we should use it or not
            if self.should_fallback_fn(self, orig_func, func, args, kwargs):
                return orig_func(*args, **kwargs)
            # torch calls inside func should be interpreted as refs calls
            with torch.overrides.enable_torch_function_mode(self, replace=self.inner):
                return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(
                f"no _refs support for {torch.overrides.resolve_name(orig_func)}"
            )
        return orig_func(*args, **kwargs)


def _is_node_supported_nvfuser(node):
    return (
        node.op == "call_function"
        and getattr(node.target, "impl_nvfuser", None) is not None
    )


def _is_func_unsupported_nvfuser(torch_function_mode, orig_func, func, args, kwargs):
    with torch.overrides.enable_torch_function_mode(
        torch_function_mode, replace=torch_function_mode.inner
    ):
        gm = get_isolated_graphmodule(func, args, kwargs)

    call_function_nodes = filter(lambda n: n.op == "call_function", gm.graph.nodes)
    any_unsupported = any(
        not _is_node_supported_nvfuser(node) for node in call_function_nodes
    )
    return any_unsupported


TorchRefsNvfuserCapabilityMode = functools.partial(
    TorchRefsMode.push, should_fallback_fn=_is_func_unsupported_nvfuser
)
