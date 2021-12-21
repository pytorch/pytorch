import torch
from .expanded_weights_impl import ExpandedWeight

def grad_if_exists(maybe_expanded_weight, per_sample_grad_fn):
    unpacked = unpack_expanded_weight_or_tensor(maybe_expanded_weight)
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        unpacked.grad_sample = per_sample_grad_fn(unpacked)

def grad_if_exists_for_input(input, grad_fn):
    if isinstance(input, torch.Tensor) and input.requires_grad:
        return grad_fn()
    else:
        return None

def unpack_expanded_weight_or_tensor(maybe_expanded_weight, func=lambda x: x):
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        orig_weight = maybe_expanded_weight.orig_weight
        return func(orig_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor) and not torch.requires_grad:
        return func(maybe_expanded_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor):
        raise RuntimeError("ExpandedWeights currently does not support a mixture of ExpandedWeight parameters "
                           "and normal Parameters. Please file and issue with pytorch/pytorch")

def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor, n_dims: int
) -> torch.Tensor:
    r"""
    Calculates the sum over all dimensions, except the first
    (batch dimension), and excluding the last n_dims.
    This function will ignore the first dimension and it will
    not aggregate over the last n_dims dimensions.
    Args:
        tensor: An input tensor of shape ``(B, ..., X[n_dims-1])``.
        n_dims: Number of dimensions to keep.
    Example:
        >>> tensor = torch.ones(1, 2, 3, 4, 5)
        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape
        torch.Size([1, 4, 5])
    Returns:
        A tensor of shape ``(B, ..., X[n_dims-1])``
    """
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)
