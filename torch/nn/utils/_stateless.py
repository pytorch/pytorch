import contextlib
from typing import Any, Callable, Optional, Dict, Tuple

import torch
from torch import Tensor
from torch.nn.utils.parametrize import ParametrizationList


def _create_new_parametrization(parametrization: ParametrizationList, tensor: Tensor) -> None:
    modules = [module for module in parametrization]
    # TODO: Find a way to cache it? but if tensor value changes each time
    # this will be hard
    return ParametrizationList(modules, tensor, parametrization.unsafe)


def _change_class(module: torch.nn.Module, apply_parametrizations: bool) -> None:
    cls = module.__class__
    func_params = module._functional_parameters
    parametrizations = None
    if apply_parametrizations:
        parametrizations = getattr(module, 'parametrizations', None)

    def _getattribute(self, name):
        if name in func_params:
            if parametrizations is not None and name in parametrizations:
                return _create_new_parametrization(parametrizations[name], func_params[name])()
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


def _swap_parameters(module: torch.nn.Module, tensor_name: str, tensor: Tensor, apply_parametrizations: bool):
    # Changes the module class to get a new __getattr__ dunder method
    # that looks for the reparametrized tensor
    if hasattr(module, "_functional_parameters"):
        module._functional_parameters[tensor_name] = tensor
    else:
        module._functional_parameters = {}
        module._functional_parameters[tensor_name] = tensor
        _change_class(module, apply_parametrizations)


def _remove_swap(module: torch.nn.Module, name: str):
    if hasattr(module, "_orig_class"):
        module.__class__ = module._orig_class
        delattr(module, "_orig_class")
        delattr(module, "_functional_parameters")


@contextlib.contextmanager
def reparametrize_module(module: torch.nn.Module, parameters_and_buffers: Dict[str, Tensor], apply_parametrizations: bool):
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            _swap_parameters,
            module, name.split("."), (tensor, apply_parametrizations))
    yield
    for name in parameters_and_buffers:
        _apply_func_submodules(
            _remove_swap,
            module, name.split("."), ())


def _apply_func_submodules(
    func: Callable[[torch.nn.Module, str, Tuple[Any]], None],
    module: torch.nn.Module,
    path: str, args: Tuple[Any],
):
    if len(path) == 1:
        func(module, path[0], *args)
    else:
        _apply_func_submodules(func, getattr(module, path[0]), path[1:], args)


def functional_call(
    module: torch.nn.Module,
    parameters_and_buffers: Dict[str, Tensor],
    args: Tuple[Any],
    kwargs : Dict[str, Any] = None,
    *, apply_parametrizations : Optional[bool] = False,
):
    r"""Performs a functional call on the module by replacing the module parameters with
    the provideed ones.

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (dict of str and Tensor): the parameters that will be used in
            the module call.
        args (tuple): arguments to be passed to the module call
        kwargs (dict): keyword arguments to be passed to the module call
        apply_parametrizations (bool): if ``True`` and ``module`` has parametrizations registered
            the parametrizations will be applied on the parameter provided in ``parameters_and_buffers``.
            If false, parametrizations will be ignored and the value in ``parameters_and_buffers`` will
            be used directly. Default: ``False``.

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
    with reparametrize_module(module, parameters_and_buffers, apply_parametrizations):
        if isinstance(args, tuple):
            out = module(*args, **kwargs)
        else:
            out = module(args, **kwargs)
    return out
