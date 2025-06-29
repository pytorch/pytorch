# mypy: allow-untyped-defs
"""Adds docstrings to functions defined in the torch.nn module."""
from torch._torch_docs import parse_kwargs


# Common parameter documentation for nn modules
common_args = parse_kwargs(
    """
    device: the device on which the parameters will be allocated. Default: None
    dtype: the data type of the parameters. Default: None
"""
)

layernorm_args = parse_kwargs(
    """
    normalized_shape: input shape from an expected input of size
        [* x normalized_shape[0] x normalized_shape[1] x ... x normalized_shape[-1]]
        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.
    eps: a value added to the denominator for numerical stability. Default: 1e-5
    elementwise_affine: a boolean value that when set to ``True``, this module
        has learnable per-element affine parameters initialized to ones (for weights)
        and zeros (for biases). Default: ``True``.
"""
)
