from typing import Dict, Union, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch._functorch.utils import exposed_in


@exposed_in("torch.func")
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
        in the ``parameters_and_buffers`` input.


         Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # does self.foo = self.foo + 1
            >>> print(mod.foo)  # tensor(0.)
            >>> functional_call(mod, a, torch.ones(()))
            >>> print(mod.foo)  # tensor(0.)
            >>> print(a['foo'])  # tensor(1.)

    An example of passing mutliple dictionaries

    .. code-block:: python

            a = ({'weight': torch.ones(1, 1)}, {'buffer': torch.zeros(1)})  # two separate dictionaries
            mod = nn.Bar(1, 1)  # return self.weight @ x + self.buffer
            print(mod.weight)  # tensor(...)
            print(mod.buffer)  # tensor(...)
            x = torch.randn((1, 1))
            print(x)
            functional_call(mod, a, x)  # same as x
            print(mod.weight)  # same as before functional_call


    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from torch.func import functional_call, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)

        def compute_loss(params, x, t):
            y = functional_call(model, params, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(dict(model.named_parameters()), x, t)

    .. note:: If the user does not need grad tracking outside of grad transforms, they can detach all of the
        parameters for better performance and memory usage

        Example::

            >>> detached_params = {k: v.detach() for k, v in model.named_parameters()}
            >>> grad_weights = grad(compute_loss)(detached_params, x, t)
            >>> grad_weights.grad_fn  # None--it's not tracking gradients outside of grad

        This means that the user cannot call ``grad_weight.backward()``. However, if they don't need autograd tracking
        outside of the transforms, this will result in less memory usage and faster speeds.

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (Dict[str,Tensor] or tuple of Dict[str, Tensor]): the parameters that will be used in
            the module call. If given a tuple of dictionaries, they must have distinct keys so that all dictionaries can
            be used together
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
        kwargs (dict): keyword arguments to be passed to the module call

    Returns:
        Any: the result of calling ``module``.
    """
    parameters_and_buffers = parameter_and_buffer_dicts if isinstance(parameter_and_buffer_dicts, dict) else {}
    if isinstance(parameter_and_buffer_dicts, tuple):
        key_list = [i for dct in parameter_and_buffer_dicts for i in dct.keys()]
        key_set = set(key_list)
        if len(key_set) != len(key_list):
            repeated_key = list(filter(lambda key: key_list.count(key) > 1, key_set))[0]
            raise ValueError(f"{repeated_key} appeared in multiple dictionaries; behavior of functional call is ambiguous")

        parameters_and_buffers = {k: v for d in parameter_and_buffer_dicts for k, v in d.items()}

    return nn.utils.stateless.functional_call(module, parameters_and_buffers, args, kwargs)
