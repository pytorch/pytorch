import torch._functorch.apis as apis
import torch._functorch.eager_transforms as _impl
import torch._functorch.make_functional as _nn_impl
from torch._functorch.vmap import in_dims_t, out_dims_t
from torch._functorch.eager_transforms import argnums_t
import torch.nn as nn
import textwrap
from typing import Any, Callable, Optional, Tuple, Union
import warnings

"""
The APIs in this file are exposed as `functorch.*`. They are thin wrappers
around the torch.func.* APIs that have deprecation warnings -- we're trying
to move people to the torch.func.* equivalents.

NB: We don't use *args, **kwargs in the signatures because that changes the
documentation.
"""

def get_warning(api, new_api=None, replace_newlines=False):
    if new_api is None:
        new_api = f'torch.func.{api}'
    warning = (
        f"We've integrated functorch into PyTorch. As the final step of the \n"
        f"integration, functorch.{api} is deprecated as of PyTorch \n"
        f"2.0 and will be deleted in a future version of PyTorch >= 2.3. \n"
        f"Please use {new_api} instead; see the PyTorch 2.0 release notes \n"
        f"and/or the torch.func migration guide for more details \n"
        f"https://pytorch.org/docs/master/func.migrating.html"
    )
    if replace_newlines:
        warning = warning.replace("\n", "")
    return warning


def warn_deprecated(api, new_api=None):
    warning = get_warning(api, new_api, replace_newlines=True)
    warnings.warn(warning, stacklevel=2)


def setup_docs(functorch_api, torch_func_api=None, new_api_name=None):
    api_name = functorch_api.__name__
    if torch_func_api is None:
        torch_func_api = getattr(_impl, api_name)
    # See https://docs.python.org/3/using/cmdline.html#cmdoption-OO
    if torch_func_api.__doc__ is None:
        return

    warning = get_warning(api_name, new_api_name)
    warning_note = "\n.. warning::\n\n" + textwrap.indent(warning, "    ")
    warning_note = textwrap.indent(warning_note, "    ")
    functorch_api.__doc__ = torch_func_api.__doc__ + warning_note

def vmap(
        func: Callable,
        in_dims: in_dims_t = 0,
        out_dims: out_dims_t = 0,
        randomness: str = 'error',
        *,
        chunk_size=None) -> Callable:
    warn_deprecated('vmap', 'torch.vmap')
    return apis.vmap(func, in_dims, out_dims, randomness, chunk_size=chunk_size)

def grad(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    warn_deprecated('grad')
    return apis.grad(func, argnums, has_aux)

def grad_and_value(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    warn_deprecated('grad_and_value')
    return _impl.grad_and_value(func, argnums, has_aux)

def vjp(func: Callable, *primals, has_aux: bool = False):
    warn_deprecated('vjp')
    return _impl.vjp(func, *primals, has_aux=has_aux)

def jvp(func: Callable, primals: Any, tangents: Any, *, strict: bool = False, has_aux: bool = False):
    warn_deprecated('jvp')
    return _impl.jvp(func, primals, tangents, strict=strict, has_aux=has_aux)

def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False,
           chunk_size: Optional[int] = None,
           _preallocate_and_copy=False):
    warn_deprecated('jacrev')
    return _impl.jacrev(func, argnums, has_aux=has_aux, chunk_size=chunk_size,
                        _preallocate_and_copy=_preallocate_and_copy)

def jacfwd(func: Callable, argnums: argnums_t = 0, has_aux: bool = False, *, randomness: str = "error"):
    warn_deprecated('jacfwd')
    return _impl.jacfwd(func, argnums, has_aux, randomness=randomness)

def hessian(func, argnums=0):
    warn_deprecated('hessian')
    return _impl.hessian(func, argnums=argnums)

def functionalize(func: Callable, *, remove: str = 'mutations') -> Callable:
    warn_deprecated('functionalize')
    return _impl.functionalize(func, remove=remove)

def make_functional(model: nn.Module, disable_autograd_tracking: bool = False):
    warn_deprecated('make_functional', 'torch.func.functional_call')
    return _nn_impl.make_functional(model, disable_autograd_tracking)

def make_functional_with_buffers(model: nn.Module, disable_autograd_tracking: bool = False):
    warn_deprecated('make_functional_with_buffers', 'torch.func.functional_call')
    return _nn_impl.make_functional_with_buffers(model, disable_autograd_tracking)

def combine_state_for_ensemble(models):
    warn_deprecated('combine_state_for_ensemble', 'torch.func.stack_module_state')
    return _nn_impl.combine_state_for_ensemble(models)

setup_docs(vmap, apis.vmap, 'torch.vmap')
setup_docs(grad, apis.grad)
setup_docs(grad_and_value)
setup_docs(vjp)
setup_docs(jvp)
setup_docs(jacrev)
setup_docs(jacfwd)
setup_docs(hessian)
setup_docs(functionalize)
setup_docs(make_functional, _nn_impl.make_functional,
           'torch.func.functional_call')
setup_docs(make_functional_with_buffers, _nn_impl.make_functional,
           'torch.func.functional_call')
setup_docs(combine_state_for_ensemble, _nn_impl.combine_state_for_ensemble,
           'torch.func.stack_module_state')
