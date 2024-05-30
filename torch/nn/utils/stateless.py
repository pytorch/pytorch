import contextlib
from collections import defaultdict
from typing import Any, Dict, Iterator, Optional, Set, Tuple, Union
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

__all__ = ["functional_call"]


def _untie_named_tensors_map(
    module: "torch.nn.Module",
    parameters_and_buffers: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """
    Unties all tied tensors in the module to parameters_and_buffers.

    This function returns a new untied_parameters_and_buffers dictionary and leave the original
    untied_parameters_and_buffers dictionary unchanged. It adds new (missing) keys for tied tensors
    in the module to untied_parameters_and_buffers. The value of the new key is the user-given value
    in the original parameters_and_buffers dictionary.

    If there are more than one user-given values for the same tied tensor, it will raise an error.

    For example, if the module has two tied weights self.foo and self.tied_foo and the user passes
    {'foo': foo_value, ...}, this will return {'foo': foo_value, 'tied_foo': foo_value, ...}. If the
    user passes {'foo': foo_value, 'tied_foo': tied_foo_value, ...}, it will raise an error. If the
    user passes {'foo': foo_value, 'tied_foo': foo_value, ...}, it will not raise an error.

    Args:
        module (torch.nn.Module): the module to determine which tensors are tied.
        parameters_and_buffers (Dict[str, Tensor]): a map of {name: tensor} for reparamaterizing the module.

    Returns:
        A new untied version of the parameters_and_buffers dictionary.

    Raises:
        ValueError: if there are more than one user-given values for the same tied tensor.
    """
    # A map of {name: tensor} for all tensors (including tied ones) in the module.
    all_named_tensors: Dict[str, Tensor] = {}
    all_named_tensors.update(module.named_parameters(remove_duplicate=False))
    all_named_tensors.update(module.named_buffers(remove_duplicate=False))

    # A map of {tensor: set(all_tied_names)} for all tensor names in the module.
    tensor_to_tied_names_map: Dict[Tensor, Set[str]] = defaultdict(set)
    for name, tensor in all_named_tensors.items():
        tensor_to_tied_names_map[tensor].add(name)

    # A map of {tied_name: set(all_tied_names)} for all tensor names in the module.
    # If a name is not tied, it will not be in this map.
    tied_names_map: Dict[str, Set[str]] = {}
    for tied_names in tensor_to_tied_names_map.values():
        if len(tied_names) > 1:
            for tied_name in tied_names:
                tied_names_map[tied_name] = tied_names

    # Make sure the user didn't pass multiple values for the same tied tensor.
    given_names = set(parameters_and_buffers.keys())
    given_names_for_tied_tensors = given_names.intersection(tied_names_map.keys())
    for given_name in given_names_for_tied_tensors:
        tied_names = tied_names_map[given_name]
        if (
            # Detect if there are multiple keys present for the same tied tensor.
            len(tied_names.intersection(given_names_for_tied_tensors)) > 1
            # Only raise an error if the user passed multiple values for the same tied tensor.
            # If all given values are the same, don't raise.
            and len({parameters_and_buffers[tied_name] for tied_name in tied_names})
            != 1
        ):
            raise ValueError(
                f"functional_call got multiple values for keys {sorted(tied_names)}, "
                f"which are tied. Consider using tie_weights=False"
            )

    # Untie the given named tensor map
    # Make a copy for not modifying the original dict
    untied_parameters_and_buffers = parameters_and_buffers.copy()
    for given_name in given_names_for_tied_tensors:
        for tied_name in tied_names_map[given_name]:
            untied_parameters_and_buffers[tied_name] = parameters_and_buffers[
                given_name
            ]
    return untied_parameters_and_buffers


@contextlib.contextmanager
def _reparametrize_module(
    module: "torch.nn.Module",
    parameters_and_buffers: Dict[str, Tensor],
    *,
    tie_weights: bool = False,
    strict: bool = False,
    stack_weights: bool = False,
) -> Iterator[None]:
    if tie_weights:
        untied_parameters_and_buffers = _untie_named_tensors_map(
            module, parameters_and_buffers
        )
    else:
        untied_parameters_and_buffers = parameters_and_buffers

    accessor = NamedMemberAccessor(module)
    if strict:
        missing_keys, unexpected_keys = accessor.check_keys(
            untied_parameters_and_buffers
        )
        error_msgs = []
        if len(unexpected_keys) > 0:
            error_msgs.append(
                f"Unexpected key(s): {', '.join(map(repr, unexpected_keys))}."
            )
        if len(missing_keys) > 0:
            error_msgs.append(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in reparametrizing for {}:\n\t{}".format(
                    module._get_name(), "\n\t".join(error_msgs)
                )
            )

    orig_parameters_and_buffers: Dict[str, Tensor] = {}
    try:
        orig_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            untied_parameters_and_buffers, allow_missing=True
        )
        yield
    finally:
        if stack_weights:
            # When stacking is enabled, we will restore the weights in LIFO order.
            orig_parameters_and_buffers = dict(
                reversed(orig_parameters_and_buffers.items())
            )
        new_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            orig_parameters_and_buffers, allow_missing=True
        )
        # Sometimes the module is not completely stateless and has some in-place modifications on
        # the _parameters and _buffers dictionaries.
        # Write the changed parameters and buffers back to the original dict.
        parameters_and_buffers.update(
            {
                k: new_parameters_and_buffers[k]
                for k in parameters_and_buffers
                if k in new_parameters_and_buffers
            }
        )


@deprecated(
    "`torch.nn.utils.stateless.functional_call` is deprecated as of PyTorch 2.0 "
    "and will be removed in a future version of PyTorch. "
    "Please use `torch.func.functional_call` instead which is a drop-in replacement.",
    category=FutureWarning,
)
def functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: Dict[str, Tensor],
    args: Union[Any, Tuple],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    r"""Perform a functional call on the module by replacing the module parameters and buffers with the provided ones.

    .. warning::

        This API is deprecated as of PyTorch 2.0 and will be removed in a future
        version of PyTorch. Please use :func:`torch.func.functional_call` instead,
        which is a drop-in replacement for this API.

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
            >>> new_a = {'foo': torch.zeros(()), 'foo_tied': torch.zeros(())}
            >>> functional_call(mod, new_a, torch.zeros()) # tensor(0.)

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (dict of str and Tensor): the parameters that will be used in
            the module call.
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
        kwargs (dict): keyword arguments to be passed to the module call
        tie_weights (bool, optional): If True, then parameters and buffers tied in the original model will be treated as
            tied in the reparamaterized version. Therefore, if True and different values are passed for the tied
            parameters and buffers, it will error. If False, it will not respect the originally tied parameters and
            buffers unless the values passed for both weights are the same. Default: True.
        strict (bool, optional): If True, then the parameters and buffers passed in must match the parameters and
            buffers in the original module. Therefore, if True and there are any missing or unexpected keys, it will
            error. Default: False.

    Returns:
        Any: the result of calling ``module``.
    """
    return _functional_call(
        module,
        parameters_and_buffers,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )


def _functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: Dict[str, Tensor],
    args: Union[Any, Tuple],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    # TODO allow kwargs such as unsafe and others for parametrization
    if (
        torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or isinstance(
            module,
            (
                torch.jit.RecursiveScriptModule,
                torch.jit.ScriptModule,
                torch.jit.ScriptFunction,
            ),
        )
    ):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    if isinstance(module, torch.nn.DataParallel):
        raise RuntimeError(
            "The stateless API can't be used with nn.DataParallel module"
        )
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    with _reparametrize_module(
        module, parameters_and_buffers, tie_weights=tie_weights, strict=strict
    ):
        return module(*args, **kwargs)
