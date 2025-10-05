"""Test for group normalization operator."""

import torch
from torchfuzz.tensor import Tensor
from .group_norm import GroupNormOperator


def test_group_norm_can_produce():
    """Test that GroupNormOperator can produce appropriate tensors."""
    op = GroupNormOperator()

    # Should not produce 1D or 2D tensors, needs at least 3D (N, C, ...)
    assert not op.can_produce(Tensor((10,), (1,), "float32", "cpu", set()))
    assert not op.can_produce(Tensor((5, 10), (10, 1), "float32", "cpu", set()))
    # Should produce 3D, 4D tensors
    assert op.can_produce(Tensor((2, 16, 32), (512, 32, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((3, 64, 5, 10), (3200, 50, 10, 1), "float32", "cpu", set()))


def test_group_norm_decompose():
    """Test group norm decomposition."""
    op = GroupNormOperator()

    # Test with minimum inputs (2) - use channels that have multiple divisors
    output_tensor = Tensor((2, 16, 32), (512, 32, 1), "float32", "cpu", set())
    inputs = op.decompose(output_tensor, num_inputs=2)

    assert len(inputs) == 2
    # Input should have same shape as output
    assert inputs[0].size == (2, 16, 32)
    assert inputs[0].dtype == "float32"
    # num_groups tensor (scalar int64)
    assert inputs[1].size == (1,)
    assert inputs[1].dtype == "int64"

    # Test with weight (3 inputs)
    inputs = op.decompose(output_tensor, num_inputs=3)
    assert len(inputs) == 3
    # Weight tensor should match number of channels
    assert inputs[2].size == (16,)  # num_channels
    assert inputs[2].dtype == "float32"

    # Test with weight and bias (4 inputs)
    inputs = op.decompose(output_tensor, num_inputs=4)
    assert len(inputs) == 4
    # Bias tensor should match number of channels
    assert inputs[3].size == (16,)  # num_channels
    assert inputs[3].dtype == "float32"


def test_group_norm_codegen():
    """Test group norm code generation."""
    op = GroupNormOperator()

    # Set up num_groups for testing
    op._num_groups = 4

    # Test with minimum inputs
    code = op.codegen("output", ["input", "num_groups"], None)
    assert code == "output = torch.nn.functional.group_norm(input, 4)"

    # Test with weight
    code = op.codegen("output", ["input", "num_groups", "weight"], None)
    assert code == "output = torch.nn.functional.group_norm(input, 4, weight=weight)"

    # Test with weight and bias
    code = op.codegen("output", ["input", "num_groups", "weight", "bias"], None)
    assert code == "output = torch.nn.functional.group_norm(input, 4, weight=weight, bias=bias)"


def test_group_norm_supports_variable_inputs():
    """Test that group norm supports variable inputs."""
    op = GroupNormOperator()
    assert op.supports_variable_inputs()


def test_group_norm_valid_groups():
    """Test that group norm creates valid group numbers."""
    op = GroupNormOperator()

    # Test with channel count that has multiple divisors
    output_tensor = Tensor((2, 32, 64), (2048, 64, 1), "float32", "cpu", set())
    inputs = op.decompose(output_tensor, num_inputs=2)

    # Verify that num_groups divides the number of channels
    num_groups = getattr(op, '_num_groups', 1)
    num_channels = 32
    assert num_channels % num_groups == 0, f"num_groups {num_groups} should divide num_channels {num_channels}"
