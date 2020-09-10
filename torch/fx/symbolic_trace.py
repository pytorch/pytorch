import inspect
from types import CodeType, FunctionType
from typing import Any, List
import torch

from .delegate import DefaultDelegate, DelegateBase, ModuleHierarchyCtxMgr
from .graph import Graph
from .graph_module import GraphModule
from .proxy import Proxy, _create_proxy
from .split import fully_outline_module

HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

def _find_module(root: torch.nn.Module, m: torch.nn.Module):
    for n, p in root.named_modules():
        if m is p:
            return n
    raise NameError('module is not installed as a submodule')

def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args : tuple
    if hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs, 0,
            0, co.co_nlocals, co.co_stacksize,
            co_flags, co.co_code, co.co_consts, co.co_names,
            co.co_varnames, co.co_filename, co.co_name,
            co.co_firstlineno, co.co_lnotab, co.co_freevars,
            co.co_cellvars
        )
    else:
        co_args = (
            nargs, 0, co.co_nlocals,
            co.co_stacksize, co_flags, co.co_code, co.co_consts,
            co.co_names, co.co_varnames, co.co_filename,
            co.co_name, co.co_firstlineno, co.co_lnotab,
            co.co_freevars, co.co_cellvars)
    new_code = CodeType(*co_args)  # type: ignore
    return FunctionType(new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)

    # we need to insert placeholder nodes for *args, and **kwargs,
    # so we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normay variables

def _proxy_placeholder(name: str, delegate: DelegateBase) -> Proxy:
    return Proxy(delegate.placeholder(name), delegate)

# Symbolic tracing API
#
# Given an `nn.Module` instance `root`, this function will return a `GraphModule`
# constructed by recording operations seen while tracing through `root`.
#
# Args:
#   - root - the `nn.Module` instance to trace
#   - delegate : An instance of a Delegate object
def symbolic_trace(root : torch.nn.Module, delegate_class=DefaultDelegate) -> GraphModule:
    graph = Graph()
    delegate = delegate_class(root, graph)

    fn = type(root).forward
    assert isinstance(fn, FunctionType)
    co = fn.__code__
    total_args = co.co_argcount + co.co_kwonlyargcount
    names_iter = iter(co.co_varnames)
    next(names_iter)  # skip self
    args : List[Any] = [root]
    args.extend(_proxy_placeholder(next(names_iter), delegate) for name in range(1, total_args))

    if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
        if co.co_flags & inspect.CO_VARARGS:
            args.append(_proxy_placeholder('*' + next(names_iter), delegate))
        if co.co_flags & inspect.CO_VARKEYWORDS:
            args.append(_proxy_placeholder('**' + next(names_iter), delegate))
        fn = _patch_function(fn, len(args))

    orig_call = torch.nn.Module.__call__

    def module_call_wrapper(mod, *args, **kwargs):
        target = _find_module(root, mod)
        if not delegate.is_leaf_module(mod):
            with ModuleHierarchyCtxMgr(target) as hier:
                return orig_call(mod, *args, **kwargs)
        else:
            return _create_proxy(delegate, 'call_module', target, args, kwargs)
    try:
        torch.nn.Module.__call__ = module_call_wrapper
        graph.output(delegate.create_arg(fn(*args)))
    finally:
        torch.nn.Module.__call__ = orig_call
    orig = GraphModule(root, graph)
    return fully_outline_module(orig)
