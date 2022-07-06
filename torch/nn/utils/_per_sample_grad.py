import functools

import torch
from torch.nn.utils._stateless import functional_call
from torch.nn.utils._expanded_weights.expanded_weights_impl import ExpandedWeight

# dependency on `functional_call` means that this can't be exposed in utils
# without creating circular dependency
from torch.utils._pytree import tree_flatten


def call_for_per_sample_grads(module, batch_size=None, loss_reduction="mean"):
    r"""
    call_for_per_sample_grads(module, batch_size, loss_reduction)
    ``call_for_per_sample_grads`` returns a function that is invoked like the forward
    function of ``module`` and will produce the same result. Then, when backward is invoked,
    the parameters of ``module`` will have a ``grad_sample`` field populated with the per sample
    gradients instead of the regular gradients

    Args:
        module: The ``nn.Module`` to get per sample gradients with respect to. All trainable
          parameters will compute per sample gradients, located in a ``grad_sample``
          field when ``backward`` is invoked
        batch_size: The batch size of the input. Typically the input's first dimension
        loss_reduction: The reduction used on the loss. Must be "mean" or "sum"
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

    def maybe_build_expanded_weight(og_tensor, batch_size):
        if og_tensor.requires_grad:
            return ExpandedWeight(og_tensor, batch_size, loss_reduction)
        else:
            return og_tensor

    def compute_batch_size(*args, **kwargs):
        args_and_kwargs = tree_flatten(args)[0] + tree_flatten(kwargs.values())[0]
        batch_size = None
        for arg in args_and_kwargs:
            if isinstance(arg, torch.Tensor):
                arg_batch_size = arg.shape[0]  # we assume batch size is the first dim
                if batch_size is not None and batch_size != arg_batch_size:
                    raise RuntimeError("When computing batch size, found at least one input with batch size "
                                       f"{batch_size} and one with batch size {arg_batch_size}. Please specify it "
                                       "explicitly using the batch size kwarg in call_for_per_sample_grads")
                batch_size = arg_batch_size
        return batch_size

    if loss_reduction not in ["sum", "mean"]:
        raise RuntimeError(f"Expected loss_reduction argument to be sum or mean, got {loss_reduction}")

    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(f"Module passed must be nn.Module, got {type(module).__name__}")
    if not (batch_size is None or isinstance(batch_size, int)):
        raise RuntimeError(f"Batch size passed must be None or an integer, got {type(batch_size).__name__}")
    if batch_size is not None and batch_size < 1:
        raise RuntimeError(f"Batch size must be positive, got {batch_size}")
    for weight in module.parameters():
        if hasattr(weight, "grad_sample") and weight.grad_sample is not None:  # type: ignore[attr-defined]
            raise RuntimeError("Current Expanded Weights accumulates the gradients, which will be incorrect for multiple "
                               f"calls without clearing gradients. Please clear out the grad_sample parameter of {weight} or "
                               "post an issue to pytorch/pytorch to prioritize correct behavior")

    @functools.wraps(module.forward)
    def wrapper(*args, **kwargs):
        nonlocal batch_size
        if batch_size is None:
            batch_size = compute_batch_size(*args, **kwargs)

        params = {name: maybe_build_expanded_weight(value, batch_size) for (name, value) in module.named_parameters()}
        return functional_call(module, params, args, kwargs)
    return wrapper
