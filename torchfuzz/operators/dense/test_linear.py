"""Test for linear operator."""

import torch
from torchfuzz.tensor import Tensor
from .linear import LinearOperator


def test_linear_can_produce():
    """Test that LinearOperator can produce appropriate tensors."""
    op = LinearOperator()

    # Should produce 1D, 2D, 3D, etc. tensors
    assert op.can_produce(Tensor((10,), (1,), "float32", "cpu", set()))
    assert op.can_produce(Tensor((5, 10), (10, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((2, 5, 10), (50, 10, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((3, 2, 5, 10), (100, 50, 10, 1), "float32", "cpu", set()))


def test_linear_decompose():
    """Test linear decomposition."""
    op = LinearOperator()

    # Test 1D output
    output_tensor = Tensor((768,), (1,), "float32", "cpu", set())
    inputs = op.decompose(output_tensor, num_inputs=2)

    assert len(inputs) == 2
    # Input should have shape (..., in_features)
    assert inputs[0].size[-1] != 768  # in_features != out_features
    # Weight should have shape (out_features, in_features)
    assert inputs[1].size[0] == 768
    assert inputs[1].size[1] == inputs[0].size[-1]

    # Test 2D output with bias
    output_tensor = Tensor((32, 512), (512, 1), "float32", "cpu", set())
    inputs = op.decompose(output_tensor, num_inputs=3)

    assert len(inputs) == 3
    # Input: (32, in_features)
    assert inputs[0].size[0] == 32
    # Weight: (512, in_features)
    assert inputs[1].size == (512, inputs[0].size[1])
    # Bias: (512,)
    assert inputs[2].size == (512,)


def test_linear_codegen():
    """Test linear code generation."""
    op = LinearOperator()

    # Test without bias
    code = op.codegen("output", ["input", "weight"], None)
    assert code == "output = torch.nn.functional.linear(input, weight)"

    # Test with bias
    code = op.codegen("output", ["input", "weight", "bias"], None)
    assert code == "output = torch.nn.functional.linear(input, weight, bias)"


def test_linear_supports_variable_inputs():
    """Test that linear supports variable inputs."""
    op = LinearOperator()
    assert op.supports_variable_inputs()
