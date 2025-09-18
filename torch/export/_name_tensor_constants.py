# Owner(s): ["oncall: export"]


import torch
from torch._subclasses.fake_tensor import FakeTensor


def set_tensor_name(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """
    Associates a custom name with a tensor for torch export.
    This is only supported for strict=False.

    When the tensor is lifted during torch export, it will use the custom name
    instead of the auto-generated name like "lifted_tensor_{N}".

    Args:
        tensor: The tensor to name
        name: The custom name to associate with the tensor

    Returns:
        The same tensor (for chaining)

    Raises:
        RuntimeError: If the tensor is a FakeTensor

    Example:
        >>> key = torch.tensor([1, 2, 3])
        >>> torch.export._name_tensor_constants.set_tensor_name(key, "my_key_tensor")
        >>> # When exported, this will be named "my_key_tensor" instead of "lifted_tensor_0"
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if isinstance(tensor, FakeTensor):
        raise RuntimeError("Cannot name FakeTensor. Only real tensors can be named.")
    if not isinstance(name, str):
        raise TypeError(f"Expected str for name, got {type(name)}")
    if not name:
        raise ValueError("Name cannot be empty")

    # Validate name contains only valid characters for FQNs
    if not all(c.isalnum() or c in "_." for c in name):
        raise ValueError(
            f"Name '{name}' contains invalid characters. Only alphanumeric, underscore, and dot are allowed."
        )

    # Store the name as a custom attribute on the tensor itself
    # This survives tensor transformations better than WeakKeyDictionary
    tensor._export_name = name  # type: ignore[attr-defined]
    return tensor
