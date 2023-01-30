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


def _create_tied_weights_map(module: 'torch.nn.Module', params_and_buffers: Dict[str, Tensor]) -> Dict[str, str]:
    """
    _create_tied_weights_map(module: Module, params_and_buffers: Dict[str, Tensor]) -> Dict[str, str]

    Creates a weight map of {tied_name: name_given_by_user} for all weights where one of their tied weights is passed

    ex: Foo() has self.foo and self.tied_foo, which are tied. If a user passed {'foo': ...} as the reparamaterization,
        this would return {'tied_foo': 'foo'}. Similarly if a user passed {'tied_foo': ...}, this returns
        {'tied_foo': 'foo'}.

    ex: If there aren't any tied weights and the user passed values for every parameter and buffer, this will return a
        map where every name maps to an empty set: {'l1.weight': set(), 'l1.bias': set(), ...}

    ex: The map only contains values that a user is reparamaterizing. For example, if module = nn.Linear(...) and the
        user only passed a new value for 'bias', this looks returns: {'bias': set()}

    This is useful because we will start by reparamaterizing all the keys of params_and_buffers, then all the key from
    this returned dictionary.
    """

    # The basic algorithm looks like:
    #   - index all weights by their original tensor value to find tied weights
    #     - when we encounter a weight not used by the user, we save it in a set (second element in the tuple)
    #     - when we run into a weight used by the user, we save that separate from the set as the first element in the tuple
    #     - ending map looks like {tensor: (name_given_by_user, set(all_tied_names)}
    #   - then loop through the values of this map (name_given_by_user and set(all_tied_names))
    #     - for each element of all_tied_names, add {tied_name: name_given_by_user} to a new map

    names = params_and_buffers.keys()
    weight_to_name_and_tied_names: Dict[torch.Tensor, Tuple[Optional[str], Set[str]]] = {}

    # create a map keyed by tensor value so that tied weights get mapped to the same key. The value is the interesting
    # part at the end it's (used_name, (tied_names)).
    # For example, in the first example where there's tied weights self.foo and self.tied_foo and the user passes a
    # value for self.foo, this will return {torch.Tensor(...): ('foo', set('tied_foo'))}
    def add_to_name_map(n: str, t: torch.Tensor):
        # if the tensor hasn't been seen before, add it to the map
        if t not in weight_to_name_and_tied_names:
            weight_to_name_and_tied_names[t] = (n, set()) if n in names else (None, {n})
            return

        # if the name is not used by the user, we add it to the tied set
        if n not in names:
            weight_to_name_and_tied_names[t][1].add(n)
            return

        # check that the user didn't pass two different tensors for the same tied weight
        first_seen_name = weight_to_name_and_tied_names[t][0]

        # if they didn't pass multiple names for tied weights or used the same tensor, we set the used name
        if first_seen_name is None or params_and_buffers[n] is params_and_buffers[first_seen_name]:
            weight_to_name_and_tied_names[t] = (n, weight_to_name_and_tied_names[t][1])
            return

        raise ValueError(f"functional_call got values for both {n} and {first_seen_name}, which are tied. " +
                         "Consider using tie_weights=False")

    tensor: Tensor
    for name, tensor in module.named_parameters(remove_duplicate=False):
        add_to_name_map(name, tensor)

    for name, tensor in module.named_buffers(remove_duplicate=False):
        add_to_name_map(name, tensor)

    # make {tied_name: name_given_by_user} from pairs of (name_given_by_user, set(all_tied_names))
    tied_weights_to_given_name = {}
    for name_given_by_user, tied_names in weight_to_name_and_tied_names.values():
        if name_given_by_user is None:  # no mapping was passed for this tensor, use original tensor
            continue
        for tied_name in tied_names:
            tied_weights_to_given_name[tied_name] = name_given_by_user
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
    *,
    tie_weights: bool = True,
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

    .. note:: If the module has tied weights, whether or not functional_call respects the tying is determined by the
        tie_weights flag.

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # has both self.foo and self.foo_tied which are tied. Returns x + self.foo + self.foo_tied
            >>> print(mod.foo)  # tensor(1.)
            >>> mod(torch.zeros(()))  # tensor(2.)
            >>> functional_call(mod, a, torch.zeros(()))  # tensor(0.) since it will change self.foo_tied too
            >>> functional_call(mod, a, torch.zeros(()), tie_weights=False)  # tensor(1.)--self.foo_tied is not updated
            >>> new_a = {'foo', torch.zeros(()), 'foo_tied': torch.zeros(())}
            >>> functional_call(mod, new_a, torch.zeros()) # tensor(0.)

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (dict of str and Tensor): the parameters that will be used in
            the module call.
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
        kwargs (dict): keyword arguments to be passed to the module call
        tie_weights (bool, optional): If True, then parameters and buffers tied in the original model will be treated as
            tied in the reparamaterized version. Therefore, if True and different values are passed for the tied
            paramaters and buffers, it will error. If False, it will not respect the originally tied parameters and
            buffers unless the values passed for both weights are the same. Default: True.

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
