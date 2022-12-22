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


@exposed_in("torch.func")
def stack_ensembled_state(models):
    """stack_ensembled_state(models) -> params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, returns two dictionaries
    that stack all of their parameters and buffers together, indexed by name.

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        def wrapper(params, buffers, data):
            return functorch.functional_call(model[0], (params, buffers), data)

        params, buffers = stack_ensembled_state(models)
        output = vmap(wrapper, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).
    """
    if len(models) == 0:
        raise RuntimeError('stack_ensembled_state: Expected at least one model, got 0.')
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError('stack_ensembled_state: Expected all models to '
                           'have the same training/eval mode.')
    model0_typ = type(models[0])
    if not all(type(m) == model0_typ for m in models):
        raise RuntimeError('stack_ensembled_state: Expected all models to '
                           'be of the same class.')
    all_params = [{k: v for k, v in model.named_parameters()} for model in models]
    params = {k: torch.stack(tuple(params[k] for params in all_params)) for k in all_params[0]}
    all_buffers = [{k: v for k, v in model.named_buffers()} for model in models]
    buffers = {k: torch.stack(tuple(buffers[k] for buffers in all_buffers)) for k in all_buffers[0]}

    return params, buffers
