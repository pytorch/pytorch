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
