"""
Polyfills for torch._C._nn functions.
"""

import torch

from ..decorators import substitute_in_graph


@substitute_in_graph(torch._C._nn._parse_to, skip_signature_check=True)
def _parse_to_polyfill(*args, **kwargs):  # noqa: F821
    """
    Polyfill for torch._C._nn._parse_to that parses arguments to nn.Module.to().

    Signature mirrors torch._C._nn._parse_to which accepts:
    - to(device) - device as string or torch.device
    - to(dtype) - dtype as torch.dtype
    - to(tensor) - extracts device and dtype from tensor
    - to(device=..., dtype=..., non_blocking=..., memory_format=...)

    Returns:
        tuple: (device, dtype, non_blocking, memory_format)
    """
    device = None
    dtype = None
    non_blocking = False
    memory_format = None

    # Handle positional arguments
    if len(args) == 1:
        arg = args[0]
        # Check if it's a tensor
        if isinstance(arg, torch.Tensor):
            device = arg.device
            dtype = arg.dtype
        # Check if it's a dtype
        elif isinstance(arg, torch.dtype):
            dtype = arg
        # Check if it's a device (string or torch.device)
        elif isinstance(arg, (str, torch.device)):
            device = torch.device(arg) if isinstance(arg, str) else arg
        else:
            raise TypeError(
                f"to() received an invalid combination of arguments. Got: {type(arg)}"
            )
    elif len(args) > 1:
        raise TypeError(
            f"to() received too many positional arguments. Got {len(args)}, expected at most 1"
        )

    # Handle keyword arguments
    if "device" in kwargs:
        device_arg = kwargs["device"]
        if device_arg is not None:
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

    if "dtype" in kwargs:
        dtype = kwargs["dtype"]

    if "non_blocking" in kwargs:
        non_blocking = kwargs["non_blocking"]

    if "memory_format" in kwargs:
        memory_format = kwargs["memory_format"]

    return (device, dtype, non_blocking, memory_format)


@substitute_in_graph(torch.__future__.get_swap_module_params_on_conversion)
def get_swap_module_params_on_conversion_polyfill() -> bool:
    """
    Polyfill for torch.__future__.get_swap_module_params_on_conversion.

    Returns the default value False to allow tracing through nn.Module._apply().
    """
    return False


@substitute_in_graph(torch._has_compatible_shallow_copy_type)
def _has_compatible_shallow_copy_type_polyfill(
    input: torch.Tensor, from_: torch.Tensor
) -> bool:
    """
    Polyfill for torch._has_compatible_shallow_copy_type.

    Checks if two tensors have compatible types for shallow copying.
    The C++ implementation checks if input's TensorImpl has compatible shallow copy type
    with from_'s key_set. We approximate this by checking if both tensors are the same type.
    """
    # Check if both tensors are the same type (handles both regular tensors and subclasses)
    # This is more permissive than checking exact torch.Tensor type equality
    # but properly handles subclasses by allowing same-type shallow copies
    return type(input) is type(from_)


__all__ = [
    "_parse_to_polyfill",
    "get_swap_module_params_on_conversion_polyfill",
    "_has_compatible_shallow_copy_type_polyfill",
]
