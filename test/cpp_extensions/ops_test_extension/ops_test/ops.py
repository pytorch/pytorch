import torch
from torch import Tensor


def test_op_with_dummy(input_tensor) -> Tensor:
    """Test function that takes a tensor and dummy, calls the op from ops.h, and returns the tensor result."""
    return torch.ops.ops_test.test_op_with_dummy.default(input_tensor)

def test_op_with_dummy_scale3(input_tensor) -> Tensor:
    """Test function that takes a tensor and dummy, calls the op from ops.h with scale=3, and returns the tensor result."""
    return torch.ops.ops_test.test_op_with_dummy_scale3.default(input_tensor)
