import torch
from .expanded_weights_impl import ExpandedWeight

def forward_helper(func, expanded_args, num_true_outs):
    r'''Forward helper computes the forward pass for a function that has expanded weight(s)
    passed to it. It will run the forward pass where all ExpandedWeights are their original
    weight. It runs checks on the given arguments and detaches the outputs.

    .. note:: First argument in :attr:`expanded_args` must be the input with the batch
    dimension as the first element of the shape

    .. note:: :attr:`func` must return a Tensor or tuple of Tensors

    Args:
        func: The function to be called
        ctx: The context from the autograd.Function object. Will be used to save
          computed state from the forward pass
        expanded_args: Arguments to be passed to :attr:`func`. Will include arguments
          that need to be unpacked because they are ExpandedWeights
        num_true_outs: The number of outputs seen by the user since some functions
          return auxillary data that is only used in the backward pass
    '''
    unexpanded_args = _check_and_unexpand_args(func, expanded_args)
    output = func(*unexpanded_args)
    output, aux_outputs = _check_and_detach_output(output, num_true_outs)
    return (output, expanded_args, aux_outputs)

def _check_and_unexpand_args(func, expanded_args):
    # input must be the first argument passed
    input = expanded_args[0]
    if isinstance(input, ExpandedWeight):
        raise RuntimeError("Expanded Weights do not support inputs that are also ExpandedWeights. "
                           f"Input must be a Tensor, got {type(input).__name__} in function {func.__name__}")
    if not isinstance(input, torch.Tensor):
        raise RuntimeError("Expanded Weights requires a Tensor as the first input to get the batch dimension, "
                           f"got {type(input).__name__} in function {func.__name__}")
    if len(input.shape) == 0:
        raise RuntimeError(f"Expanded Weights requires a batch dimension but got an input of size 0 in function {func.__name__}")
    if input.shape[0] == 0:
        raise RuntimeError("0 is not a valid batch size for Expanded Weights but got input tensor of "
                           f"{input} in function {func.__name__}")
    batch_size = input.shape[0]
    for arg in expanded_args:
        if isinstance(arg, ExpandedWeight) and arg.batch_size != batch_size:
            raise RuntimeError("Expected ExpandedWeights to have batch size matching input but got "
                               f"input batch size of {batch_size} with ExpandedWeight of batch size {arg.batch_size}")

    unexpanded_args = tuple(arg.orig_weight if isinstance(arg, ExpandedWeight) else arg for arg in expanded_args)
    return unexpanded_args

def _check_and_detach_output(output, num_true_outs):
    aux_outputs = None
    # separates differentiable outputs from outputs only needed for the backwards computation
    if isinstance(output, tuple):
        if len(output) < num_true_outs:
            raise RuntimeError(f"Got fewer outputs ({len(output)}) than expected ({num_true_outs}). "
                               "Issues in ExpandedWeights' autograd.Function")
        aux_outputs = output[num_true_outs:]
        if num_true_outs == 1:
            output = output[0]
        else:
            output = output[:num_true_outs]
    elif num_true_outs != 1:
        raise RuntimeError(f"Got single output but expected at least {num_true_outs} outputs. "
                           "Issues in ExpandedWeights' autograd.Function")
    return (output, aux_outputs)

def grad_if_exists(maybe_expanded_weight, per_sample_grad_fn):
    unpacked = unpack_expanded_weight_or_tensor(maybe_expanded_weight)
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        if hasattr(unpacked, "grad_sample"):
            if isinstance(unpacked.grad_sample, list):
                unpacked.grad_sample.append(per_sample_grad_fn(unpacked))
            else:
                unpacked.grad_sample = [unpacked.grad_sample, per_sample_grad_fn(unpacked)]
        else:
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
    elif isinstance(maybe_expanded_weight, torch.Tensor) and not maybe_expanded_weight.requires_grad:
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
