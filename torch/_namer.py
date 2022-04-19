import torch
import types
import torch.overrides
import torch.fx.node
import functools
import __future__

# copied from torch.overrides
@functools.lru_cache(None)
def get_overridable_functions_index():
    """List functions that are overridable via __torch_function__

    Returns
    -------
    Dict[Any, List[Callable]]
        A dictionary that maps namespaces that contain overridable functions
        to functions in that namespace that can be overridden.
    """
    index = {}
    tested_namespaces = [
        ("torch._C._VariableFunctions", dir(torch._C._VariableFunctions)),
        ("torch._C._nn", dir(torch._C._nn)),
        ("torch", torch.__all__ + dir(torch._C._VariableFunctions)),
        ("torch.functional", torch.functional.__all__),
        ("torch.nn.functional", dir(torch.nn.functional)),
        ("torch.nn.init", dir(torch.nn.init)),
        ("torch.Tensor", dir(torch.Tensor)),
        ("torch.linalg", dir(torch.linalg)),
        ("torch.fft", dir(torch.fft)),
        ("torch.special", dir(torch.special)),
    ]
    for namespace_str, ns_funcs in tested_namespaces:
        namespace = eval(namespace_str)
        for func_name in ns_funcs:
            # ignore private functions or functions that are deleted in torch.__init__
            if namespace is not torch.Tensor:
                # TODO: I deleted the private stuff, this is bad news, it
                # means we are hitting private API here
                if func_name.startswith('__'):
                    continue
                if func_name == 'unique_dim':
                    continue
            else:
                func = getattr(namespace, func_name)
                if getattr(object, func_name, None) == func:
                    continue
                if func_name == '__weakref__':
                    continue
            func = getattr(namespace, func_name)
            if namespace is torch.Tensor and getattr(object, func_name, None) == func:
                continue
            # ignore re-exported modules
            if isinstance(func, types.ModuleType):
                continue
            # ignore __future__ imports
            if isinstance(func, __future__._Feature):
                continue

            if not callable(func) and hasattr(func, "__get__"):
                index[func.__get__] = f"{namespace_str}.{func_name}.__get__"
                continue

            if not callable(func):
                continue

            index[func] = f"{namespace_str}.{func_name}"
    return index

def resolve_func(func):
    r = get_overridable_functions_index().get(func, None)
    if r is not None:
        return r
    elif hasattr(func, '__module__') and hasattr(func, '__name__'):
        return f'{func.__module__}.{func.__name__}'
    else:
        return None
