import contextlib
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union, Set, Optional

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


def _create_tied_weights_map(module, params_and_buffers):
    # creates a weight map of {tied_name: name_given_by_user} for all weights where one of their tied weights is passed
    #
    # The basic algorithm looks like:
    #   - index all weights by their original tensor value to find tied weights
    #     - when we encounter a weight not used by the user, we save it in a set (second element in the tuple)
    #     - when we run into a weight used by the user, we save that separate from the set as the first element in the tuple
    #     - ending map looks like {tensor: (name_given_by_user, set(all_tied_names)}
    #   - then loop through the values of this map (name_given_by_user and set(all_tied_names))
    #     - for each element of all_tied_names, add {tied_name: name_given_by_user} to a new map

    names = params_and_buffers.keys()
    weight_to_name_and_tied_names: Dict[torch.Tensor, Tuple[Optional[str], Set[str]]] = {}

    def add_to_name_map(name, t):
        if t in weight_to_name_and_tied_names:
            first_seen_name = weight_to_name_and_tied_names[t][0]
            if name in names and first_seen_name and params_and_buffers[name] is not params_and_buffers[first_seen_name]:
                raise ValueError(f"functional_call got values for both {name} and {first_seen_name}, which are tied.")
            elif name in names:
                weight_to_name_and_tied_names[t] = (name, weight_to_name_and_tied_names[t][1])
            else:
                weight_to_name_and_tied_names[t][1].add(name)
        else:
            weight_to_name_and_tied_names[t] = (name, set()) if name in names else (None, {name})

    for name, t in module.named_parameters(remove_duplicate=False):
        add_to_name_map(name, t)

    for name, t in module.named_buffers(remove_duplicate=False):
        add_to_name_map(name, t)

    # make {tied_name: name_given_by_user} from pairs of (name_given_by_user, set(all_tied_names))
    tied_weights_to_given_name = {}
    for name, tied_names in weight_to_name_and_tied_names.values():
        if name is None:  # no mapping was passed for this tensor, use original tensor
            continue
        for tied_name in tied_names:
            tied_weights_to_given_name[tied_name] = name
    return tied_weights_to_given_name


def _create_swap_params(params_and_buffers):
    def _swap_parameters(module, tensor_name: str, full_path: str, tensor: Optional[Tensor]) -> None:
        # Changes the module class to get a new __getattr__ dunder method
        # that looks for the reparametrized tensor
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
    tie_weights: bool = False,
) -> Iterator[None]:
    tied_weights_map = _create_tied_weights_map(module, parameters_and_buffers) if tie_weights else {}
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            _create_swap_params(parameters_and_buffers),
            module, name.split("."), name, (tensor,))
    for tied_name, user_given_name in tied_weights_map.items():
        _apply_func_submodules(
            _create_swap_params(parameters_and_buffers),
            module, tied_name.split("."), user_given_name, (None,))
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
    args: Union[Any, Tuple],
    kwargs: Dict[str, Any] = None,
    tie_weights: bool = False,
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
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
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
    with _reparametrize_module(module, parameters_and_buffers, tie_weights):
        if isinstance(args, tuple):
            out = module(*args, **kwargs)
        else:
            out = module(args, **kwargs)
    return out
