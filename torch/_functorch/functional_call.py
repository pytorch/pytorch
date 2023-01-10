from typing import Dict, Union, Any, Tuple, List

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
        in the ``parameters_and_buffers`` input.


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
        tie_weights (bool, optional): If True, then parameters and buffers tied in the original model will be treated as
            tied in the reparamaterized version. Therefore, if True and different values are passed for the tied
            paramaters and buffers, it will error. If False, it will not respect the originally tied parameters and
            buffers unless the values passed for both weights are the same. Default: True.

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

    return nn.utils.stateless.functional_call(module, parameters_and_buffers, args, kwargs, tie_weights=tie_weights)


@exposed_in("torch.func")
def stack_module_state(models: List[nn.Module]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """stack_module_state(models) -> params, buffers

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
            return torch.func.functional_call(model[0], (params, buffers), data)

        params, buffers = stack_module_state(models)
        output = vmap(wrapper, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    When there's submodules, this follows state dict naming conventions

    .. code-block:: python

        import torch.nn as nn
        class Foo(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                hidden = 4
                self.l1 = nn.Linear(in_features, hidden)
                self.l2 = nn.Linear(hidden, out_features)

            def forward(self, x):
                return self.l2(self.l1(x))

        num_models = 5
        in_features, out_features = 3, 3
        models = [Foo(in_features, out_features) for i in range(num_models)]
        params, buffers = stack_module_state(models)
        print(list(params.keys()))  # "l1.weight", "l1.bias", "l2.weight", "l2.bias"

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).
    """
    if len(models) == 0:
        raise RuntimeError('stack_module_state: Expected at least one model, got 0.')
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError('stack_module_state: Expected all models to '
                           'have the same training/eval mode.')
    model0_typ = type(models[0])
    if not all(type(m) == model0_typ for m in models):
        raise RuntimeError('stack_module_state: Expected all models to '
                           'be of the same class.')
    all_params = [{k: v for k, v in model.named_parameters()} for model in models]
    params = {k: torch.stack(tuple(params[k] for params in all_params)) for k in all_params[0]}
    all_buffers = [{k: v for k, v in model.named_buffers()} for model in models]
    buffers = {k: torch.stack(tuple(buffers[k] for buffers in all_buffers)) for k in all_buffers[0]}

    return params, buffers
