import contextlib

import torch
from torch._expanded_weights.expanded_weights_impl import ExpandedWeight

@contextlib.contextmanager
def reparametrize_module(module, parameters_and_buffers):
    # Parametrization does not support to change submodules directly
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            torch.nn.utils.parametrize.register_parametrization,
            module, name.split("."), (_ReparametrizedTensor(tensor),))
    yield
    for name in parameters_and_buffers:
        _apply_func_submodules(
            torch.nn.utils.parametrize.remove_parametrizations,
            module, name.split("."), (False,))


class _ReparametrizedTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self._tensor = tensor

    def forward(self, original):
        return self._tensor


def _apply_func_submodules(func, module, path, args):
    if len(path) == 1:
        func(module, path[0], *args)
    else:
        _apply_func_submodules(func, getattr(module, path[0]), path[1:], args)


def functional_call(module, parameters_and_buffers, args, kwargs=None):
    # TODO allow kwargs such as unsafe and others for parametrization
    if kwargs is None:
        kwargs = {}
    with reparametrize_module(module, parameters_and_buffers):
        if isinstance(args, tuple):
            out = module(*args, **kwargs)
        else:
            out = module(args, **kwargs)
    return out

def per_sample_call(module, batch_size, args, kwargs=None):
    r"""
    per_sample_call(module, batch_size, args, kwargs=None) -> Tensor

    Invoked just like a forward pass, ``per_sample_call`` will produce the same
    forward result. Then, when backward is invoked, the parameters of ``module``
    will have a ``grad_sample`` field populated with the per sample gradients
    instead of the batched gradients

    Args:
        module: The module to get per sample gradients with respect to. All trainable
          parameters will compute per sample gradients, located in a ``grad_sample``
          field when ``backward`` is invoked
        batch_size: The batch size of the input. Typically the input's first dimension
        args: Tuple of positional args passed to ``module`` to perform the forward pass
        kwargs: Dict of named args passed to ``module`` to perform the forward pass.
          Default: None

    Examples::

        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = per_sample_call(model, batched_input.shape[0], batched_input).sum()
        >>> res.backward()
        >>> assert model.weight.shape == (3, 4)
        >>> assert model.weight.grad_sample.shape == (5, 3, 4)
        >>> assert model.weight.grad == None
        >>> assert model.bias.shape == (3,)
        >>> assert model.bias.grad_sample.shape == (5, 3)
        >>> assert model.bias.grad == None
    """
    def maybe_build_expanded_weight(og_tensor):
        if og_tensor.requires_grad:
            return ExpandedWeight(og_tensor, batch_size)
        else:
            return og_tensor

    params = {name: maybe_build_expanded_weight(value) for (name, value) in module.named_parameters()}
    return functional_call(module, params, args, kwargs)
