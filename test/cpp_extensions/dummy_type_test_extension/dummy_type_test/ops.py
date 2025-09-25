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
