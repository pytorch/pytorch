import torch
from torch.nn.utils._stateless import functional_call
from torch.nn.utils._expanded_weights.expanded_weights_impl import ExpandedWeight

# dependency on `functional_call` means that this can't be exposed in utils
# without creating circular dependency
def call_for_per_sample_grads(module, batch_size, args, kwargs=None):
    r"""
    call_for_per_sample_grads(module, batch_size, args, kwargs=None) -> Tensor
    Invoked just like a forward pass, ``call_for_per_sample_grads`` will produce the same
    forward result. Then, when backward is invoked, the parameters of ``module``
    will have a ``grad_sample`` field populated with the per sample gradients
    instead of the regular gradients

    Args:
        module: The ``nn.Module`` to get per sample gradients with respect to. All trainable
          parameters will compute per sample gradients, located in a ``grad_sample``
          field when ``backward`` is invoked
        batch_size: The batch size of the input. Typically the input's first dimension
        args: Tuple of positional args passed to ``module`` to perform the forward pass
        kwargs: Dict of named args passed to ``module`` to perform the forward pass. Default: None

    Examples::
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = call_for_per_sample_grads(model, batched_input.shape[0], batched_input).sum()
        >>> res.backward()
        >>> assert model.weight.shape == (3, 4)
        >>> assert model.weight.grad_sample.shape == (5, 3, 4)
        >>> assert model.weight.grad == None
        >>> assert model.bias.shape == (3,)
        >>> assert model.bias.grad_sample.shape == (5, 3)
        >>> assert model.bias.grad == None

    Note::
        Does not work with any `nn.RNN`, including `nn.GRU` or `nn.LSTM`. Please use custom
        rewrites that wrap an `nn.Linear` module. See Opacus for an example
    """
    def maybe_build_expanded_weight(og_tensor):
        if og_tensor.requires_grad:
            return ExpandedWeight(og_tensor, batch_size)
        else:
            return og_tensor

    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(f"Module passed must be nn.Module, got {type(module).__name__}")
    if not isinstance(batch_size, int):
        raise RuntimeError(f"Batch size passed must be an integer, got {type(batch_size).__name__}")
    if batch_size < 1:
        raise RuntimeError(f"Batch size must be positive, got {batch_size}")
    for weight in module.parameters():
        if hasattr(weight, "grad_sample") and weight.grad_sample is not None:  # type: ignore[attr-defined]
            raise RuntimeError("Current Expanded Weights accumulates the gradients, which will be incorrect for multiple "
                               f"calls without clearing gradients. Please clear out the grad_sample parameter of {weight} or "
                               "post an issue to pytorch/pytorch to prioritize correct behavior")
    params = {name: maybe_build_expanded_weight(value) for (name, value) in module.named_parameters()}
    return functional_call(module, params, args, kwargs)
