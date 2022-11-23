import warnings
import contextlib
from typing import Any, Callable, Dict, Iterator, List, Tuple

import torch
from torch import Tensor

__all__ = ["functional_call"]

# We avoid typing module here because module attributes are declared as Union[Parameter, Tensor] by default
# and using other types causes mypy errors
def _change_class(module, params_and_buffers) -> None:
    cls = module.__class__
    attr_to_path : Dict[str, str] = module._attr_to_path

    def _getattribute(self, name: str) -> Any:
        if name in attr_to_path:
            return params_and_buffers[attr_to_path[name]]
        return cls.__getattribute__(self, name)

    def _setattr(self, name: str, value: Any) -> None:
        if name in attr_to_path:
            params_and_buffers[attr_to_path[name]] = value
        else:
            return cls.__setattr__(self, name, value)

    param_cls = type(
        f"StatelessReplacer{cls.__name__}",
        (cls,),
        {
            "__getattribute__": _getattribute,
            "__setattr__": _setattr,
        },
    )

    module.__class__ = param_cls
    module._orig_class = cls


def _check_tied_val_already_replaced(old_val, new_val, replaced_tensors_map):
    if old_val not in replaced_tensors_map:
        replaced_tensors_map[old_val] = new_val
    elif replaced_tensors_map[old_val] is not new_val:
        warnings.warn("functional_call was passed multiple values for tied weights. "
                      "This behavior is deprecated and will be an error in future versions")


def _create_swap_params(params_and_buffers, replaced_tensors_map):
    def _swap_parameters(module, tensor_name: str, full_path: str, tensor: Tensor) -> None:
        # Changes the module class to get a new __getattr__ dunder method
        # that looks for the reparametrized tensor
        if hasattr(module, tensor_name):
            old_val = getattr(module, tensor_name)
            _check_tied_val_already_replaced(old_val, tensor, replaced_tensors_map)
        if hasattr(module, "_attr_to_path"):
            module._attr_to_path[tensor_name] = full_path
        else:
            module._attr_to_path = {}
            module._attr_to_path[tensor_name] = full_path
            _change_class(module, params_and_buffers)
    return _swap_parameters


def _remove_swap(module, name: str, full_path: str) -> None:
    if hasattr(module, "_orig_class"):
        module.__class__ = module._orig_class
        delattr(module, "_orig_class")
        delattr(module, "_attr_to_path")


@contextlib.contextmanager
def _reparametrize_module(
    module: 'torch.nn.Module',
    parameters_and_buffers: Dict[str, Tensor],
) -> Iterator[None]:
    orig_tensors_to_replacements: Dict[Tensor, Tensor] = {}
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            _create_swap_params(parameters_and_buffers, orig_tensors_to_replacements),
            module, name.split("."), name, (tensor,))
    try:
        yield
    finally:
        for name in parameters_and_buffers:
            _apply_func_submodules(
                _remove_swap,
                module, name.split("."), name, ())


def _apply_func_submodules(
    func: Callable[..., None],
    module: 'torch.nn.Module',
    path: List[str],
    full_path: str,
    args: Tuple,
):
    if len(path) == 1:
        func(module, path[0], full_path, *args)
    else:
        _apply_func_submodules(func, getattr(module, path[0]), path[1:], full_path, args)


def functional_call(
    module: 'torch.nn.Module',
    parameters_and_buffers: Dict[str, Tensor],
    args: Tuple,
    kwargs : Dict[str, Any] = None,
):
    r"""Performs a functional call on the module by replacing the module parameters
    and buffers with the provided ones.

    .. note:: If the module has active parametrizations, passing a value in the
        :attr:`parameters_and_buffers` argument with the name set to the regular parameter
        name will completely disable the parametrization.
        If you want to apply the parametrization function to the value passed
        please set the key as ``{submodule_name}.parametrizations.{parameter_name}.original``.

    .. note:: If the module performs in-place operations on parameters/buffers, these will be reflected
        in the `parameters_and_buffers` input.

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # does self.foo = self.foo + 1
            >>> print(mod.foo)  # tensor(0.)
            >>> functional_call(mod, a, torch.ones(()))
            >>> print(mod.foo)  # tensor(0.)
            >>> print(a['foo'])  # tensor(1.)

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
    with _reparametrize_module(module, parameters_and_buffers):
        if isinstance(args, tuple):
            out = module(*args, **kwargs)
        else:
            out = module(args, **kwargs)
    return out
