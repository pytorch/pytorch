import torch
from torch import Tensor


def test_fn(tensor: Tensor, dummy_type_instance) -> Tensor:
    """
    Test function that demonstrates version-aware DummyType conversions.

    This function takes a tensor and a DummyType instance, extracts the id
    from the DummyType, and fills the tensor with that id value.

    Args:
        tensor: Input tensor to be filled
        dummy_type_instance: An instance of DummyType (version depends on compile-time targeting)

    Returns:
        Tensor filled with the id value from the DummyType instance
    """
    return torch.ops.dummy_type_test.test_fn.default(tensor, dummy_type_instance)


def create_dummy(tensor: Tensor):
    """
    Create a DummyType instance with id 42.

    This function takes a tensor as input and returns a DummyType instance
    with a fixed id value of 42. The tensor input is not used in the computation
    but is required by the function signature.

    Args:
        tensor: Input tensor (not used in computation, but required by signature)

    Returns:
        DummyType instance with id=42
    """
    return torch.ops.dummy_type_test.create_dummy.default(tensor)
