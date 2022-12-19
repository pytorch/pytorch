from typing import Dict, Union, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def functional_call(
    module: 'torch.nn.Module',
    parameter_and_buffer_dicts: Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], ...]],
    args: Union[Any, Tuple],
    kwargs: Dict[str, Any] = None,
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

    .. note:: If the user does not need grad tracking outside of grad transforms, they can detach all of the
        parameters for better performance and memory usage

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # does self.foo = self.foo + 1
            >>> print(mod.foo)  # tensor(0.)
            >>> functional_call(mod, a, torch.ones(()))
            >>> print(mod.foo)  # tensor(0.)
            >>> print(a['foo'])  # tensor(1.)

        Example::

            >>> a = ({'weight': torch.ones(1, 1)}, {'buffer': torch.zeros(1)})  # two separate dictionaries
            >>> # xdoctest: +SKIP
            >>> mod = nn.Bar(1, 1)  # return self.weight @ x + self.buffer
            >>> print(mod.weight)  # tensor(...)
            >>> print(mod.buffer)  # tensor(...)
            >>> x = torch.randn((1, 1))
            >>> print(x)
            >>> functional_call(mod, a, x)  # same as x
            >>> print(mod.weight)  # same as before functional_call

        Example::

            >>> mod = nn.Linear(1, 1)
            >>> d = {k: v.detach() for k, v in mod.named_parameters()}
            >>> grad(lambda x: functional_call(mod, d, x), torch.randn((1, 1)))  # doesn't tracks grads for params

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (dict of str and Tensor or tuple of str and Tensors): the parameters that will be used in
            the module call. If given a tuple of dictionaries, they must have distinct keys so that all dictionaries can
            be used together
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
        kwargs (dict): keyword arguments to be passed to the module call

    Returns:
        Any: the result of calling ``module``.
    """
    parameters_and_buffers = parameter_and_buffer_dicts if isinstance(parameter_and_buffer_dicts, dict) else {}
    if isinstance(parameter_and_buffer_dicts, tuple):
        keys = [parameter_and_buffer.keys() for parameter_and_buffer in parameter_and_buffer_dicts]
        for key in keys:
            if keys.count(key) > 1:
                raise ValueError(f"{key} appeared in multiple dictionaries; behavior of functional call is ambiguous")

        parameters_and_buffers = {k: v for d in parameter_and_buffer_dicts for k, v in d.items()}

    return nn.utils.stateless.functional_call(module, parameters_and_buffers, args, kwargs)
