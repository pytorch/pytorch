import contextlib
from typing import Any, Callable, Dict, Iterator, List, Tuple

import torch
from torch import Tensor


# We avoid typing module here because module attributes are declared as Union[Parameter, Tensor] by default
# and using other types causes mypy errors
def _change_class(module) -> None:
    cls = module.__class__
    func_params : Dict[str, Tensor] = module._functional_parameters

    def _getattribute(self, name: str) -> Any:
        if name in func_params:
            return func_params[name]
        return cls.__getattribute__(self, name)

    param_cls = type(
        f"StatelessReplacer{cls.__name__}",
        (cls,),
        {
            "__getattribute__": _getattribute,
        },
    )

    module.__class__ = param_cls
    module._orig_class = cls


def _swap_parameters(module, tensor_name: str, tensor: Tensor) -> None:
    # Changes the module class to get a new __getattr__ dunder method
    # that looks for the reparametrized tensor
    if hasattr(module, "_functional_parameters"):
        module._functional_parameters[tensor_name] = tensor
    else:
        module._functional_parameters = {}
        module._functional_parameters[tensor_name] = tensor
        _change_class(module)


def _remove_swap(module, name: str) -> None:
    if hasattr(module, "_orig_class"):
        module.__class__ = module._orig_class
        delattr(module, "_orig_class")
        delattr(module, "_functional_parameters")


@contextlib.contextmanager
def reparametrize_module(
    module: torch.nn.Module,
    parameters_and_buffers: Dict[str, Tensor],
) -> Iterator[None]:
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            _swap_parameters,
            module, name.split("."), (tensor,))
    yield
    for name in parameters_and_buffers:
        _apply_func_submodules(
            _remove_swap,
            module, name.split("."), ())


def _apply_func_submodules(
    func: Callable[..., None],
    module: torch.nn.Module,
    path: List[str],
    args: Tuple,
):
    if len(path) == 1:
        func(module, path[0], *args)
    else:
        _apply_func_submodules(func, getattr(module, path[0]), path[1:], args)


def functional_call(
    module: torch.nn.Module,
    parameters_and_buffers: Dict[str, Tensor],
    args: Tuple,
    kwargs : Dict[str, Any] = None,
):
    r"""Performs a functional call on the module by replacing the module parameters
    and buffers with the provided ones.

    .. note:: If the module has active parametrizations, passing a value in the
        `parameters_and_buffers` argument with the name set to the regular parameter
         name will completely disable the parametrization.
         If you want to apply the parametrization function to the value passed
         please set the key as `{submodule_name}.parametrizations.{parameter_name}.original`.

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (dict of str and Tensor): the parameters that will be used in
            the module call.
        args (tuple): arguments to be passed to the module call
        kwargs (dict): keyword arguments to be passed to the module call

    Returns:
        Any: the result of calling ``module``.
    """
    # TODO allow kwargs such as unsafe and others for parametrization
    if (
            torch.jit.is_tracing()
            or torch.jit.is_scripting()
            or isinstance(module, (
                torch.jit.RecursiveScriptModule,
                torch.jit.ScriptModule,
                torch.jit.ScriptFunction)
            )
    ):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    if kwargs is None:
        kwargs = {}
    with reparametrize_module(module, parameters_and_buffers):
        if isinstance(args, tuple):
            out = module(*args, **kwargs)
        else:
            out = module(args, **kwargs)
    return out
