# type: ignore
import inspect
from types import CodeType, FunctionType
import torch

from .graph import Graph
from .graph_module import GraphModule

HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

def _find_module(root, m):
    for n, p in root.named_modules():
        if m is p:
            return n
    raise NameError('module is not installed as a submodule')

def _patch_function(fn, nargs):
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
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
    new_code = CodeType(*co_args)
    return FunctionType(new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)

    # we need to insert placeholder nodes for *args, and **kwargs,
    # so we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normay variables

def is_leaf_module(m):
    return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

def symbolic_trace(root, is_leaf_module=is_leaf_module):
    def _use_parameter(graph, a):
        if isinstance(a, torch.nn.Parameter):
            for n, p in root.named_parameters():
                if a is p:
                    return graph.get_param(n)
            raise NameError('parameter is not a member of this module')
        return NotImplemented
    graph = Graph(arg_handler=_use_parameter)
    fn = type(root).forward

    co = fn.__code__
    total_args = co.co_argcount + co.co_kwonlyargcount
    names_iter = iter(co.co_varnames)
    next(names_iter)  # skip self
    args = [root]
    args.extend(graph.placeholder(next(names_iter)) for name in range(1, total_args))

    if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
        if co.co_flags & inspect.CO_VARARGS:
            args.append(graph.placeholder('*' + next(names_iter)))
        if co.co_flags & inspect.CO_VARKEYWORDS:
            args.append(graph.placeholder('**' + next(names_iter)))
        fn = _patch_function(fn, len(args))

    args = tuple(args)
    orig_call = torch.nn.Module.__call__

    def module_call_wrapper(mod, *args, **kwargs):
        if not is_leaf_module(mod):
            return orig_call(mod, *args, **kwargs)
        else:
            target = _find_module(root, mod)
            return graph.call_module(target, args, kwargs)
    try:
        torch.nn.Module.__call__ = module_call_wrapper
        graph.output(fn(*args))
    finally:
        torch.nn.Module.__call__ = orig_call
    return GraphModule(root, graph)
