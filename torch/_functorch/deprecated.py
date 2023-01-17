import torch._functorch.vmap as _vmap_impl
import torch._functorch.eager_transforms as _impl
import torch._functorch.make_functional as _nn_impl
import textwrap
import warnings

"""
The APIs in this file are exposed as `functorch.*`. They are thin wrappers
around the torch.func.* APIs that have deprecation warnings -- we're trying
to move people to the torch.func.* equivalents.
"""

def get_warning(api, new_api=None, replace_newlines=False):
    if new_api is None:
        new_api = f'torch.func.{api}'
    warning = (
        f"We've integrated functorch into PyTorch. As the final step of the \n"
        f"integration, functorch.{api} is deprecated as of PyTorch \n"
        f"2.0 and will be deleted in a future version of PyTorch >= 2.3. \n"
        f"Please use {new_api} instead; see the PyTorch 2.0 release notes \n"
        f"and/or the torch.func migration guide for more details."
    )
    if replace_newlines:
        warning = warning.replace("\n", "")
    return warning


def warn_deprecated(api, new_api=None):
    warning = get_warning(api, new_api, replace_newlines=True)
    warnings.warn(warning)


def setup_docs_annotations(functorch_api, torch_func_api=None, new_api_name=None):
    api_name = functorch_api.__name__
    if torch_func_api is None:
        torch_func_api = getattr(_impl, api_name)

    functorch_api.__annotations__ = torch_func_api.__annotations__

    warning = get_warning(api_name, new_api_name)
    warning_note = "\n.. warning::\n\n" + textwrap.indent(warning, "    ")
    warning_note = textwrap.indent(warning_note, "    ")
    functorch_api.__doc__ = torch_func_api.__doc__ + warning_note

def vmap(*args, **kwargs):
    warn_deprecated('vmap', 'torch.vmap')
    return _vmap_impl.vmap(*args, **kwargs)

def grad(*args, **kwargs):
    warn_deprecated('grad')
    return _impl.grad(*args, **kwargs)

def grad_and_value(*args, **kwargs):
    warn_deprecated('grad_and_value')
    return _impl.grad_and_value(*args, **kwargs)

def vjp(*args, **kwargs):
    warn_deprecated('vjp')
    return _impl.vjp(*args, **kwargs)

def jvp(*args, **kwargs):
    warn_deprecated('jvp')
    return _impl.jvp(*args, **kwargs)

def jacrev(*args, **kwargs):
    warn_deprecated('jacrev')
    return _impl.jacrev(*args, **kwargs)

def jacfwd(*args, **kwargs):
    warn_deprecated('jacfwd')
    return _impl.jacfwd(*args, **kwargs)

def hessian(*args, **kwargs):
    warn_deprecated('hessian')
    return _impl.hessian(*args, **kwargs)

def functionalize(*args, **kwargs):
    warn_deprecated('functionalize')
    return _impl.functionalize(*args, **kwargs)

def make_functional(*args, **kwargs):
    warn_deprecated('make_functional', 'torch.func.functional_call')
    return _nn_impl.make_functional(*args, **kwargs)

def make_functional_with_buffers(*args, **kwargs):
    warn_deprecated('make_functional_with_buffers', 'torch.func.functional_call')
    return _nn_impl.make_functional_with_buffers(*args, **kwargs)

def combine_state_for_ensemble(*args, **kwargs):
    warn_deprecated('combine_state_for_ensemble', 'torch.func.stack_module_state')
    return _nn_impl.combine_state_for_ensemble(*args, **kwargs)

setup_docs_annotations(vmap, _vmap_impl.vmap, 'torch.vmap')
setup_docs_annotations(grad)
setup_docs_annotations(grad_and_value)
setup_docs_annotations(vjp)
setup_docs_annotations(jvp)
setup_docs_annotations(jacrev)
setup_docs_annotations(jacfwd)
setup_docs_annotations(hessian)
setup_docs_annotations(functionalize)
setup_docs_annotations(make_functional, _nn_impl.make_functional,
                       'torch.func.functional_call')
setup_docs_annotations(make_functional_with_buffers, _nn_impl.make_functional,
                       'torch.func.functional_call')
setup_docs_annotations(combine_state_for_ensemble, _nn_impl.combine_state_for_ensemble,
                       'torch.func.stack_module_state')
