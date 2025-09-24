"""Test for layer normalization operator."""

import torch
from torchfuzz.tensor import Tensor
from .layer_norm import LayerNormOperator


def test_layer_norm_can_produce():
    """Test that LayerNormOperator can produce appropriate tensors."""
    op = LayerNormOperator()

    # Should not produce 1D tensors, needs at least 2D
    assert not op.can_produce(Tensor((10,), (1,), "float32", "cpu", set()))
    # Should produce 2D, 3D, 4D tensors
    assert op.can_produce(Tensor((5, 10), (10, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((2, 5, 768), (3840, 768, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((3, 2, 5, 10), (100, 50, 10, 1), "float32", "cpu", set()))


def test_layer_norm_decompose():
    """Test layer norm decomposition."""
    op = LayerNormOperator()

    # Test with minimum inputs (2)
    output_tensor = Tensor((32, 768), (768, 1), "float32", "cpu", set())
    inputs = op.decompose(output_tensor, num_inputs=2)

    assert len(inputs) == 2
    # Input should have same shape as output
    assert inputs[0].size == (32, 768)
    assert inputs[0].dtype == "float32"
    # Shape tensor (representing normalized_shape dimensions)
    assert inputs[1].dtype == "int64"
    assert len(inputs[1].size) == 1  # 1D tensor representing shape

    # Test with weight (3 inputs)
    inputs = op.decompose(output_tensor, num_inputs=3)
    assert len(inputs) == 3
    # Weight tensor should match normalized_shape
    normalized_shape = getattr(op, '_normalized_shape', None)
    if normalized_shape:
        assert inputs[2].size == normalized_shape
        assert inputs[2].dtype == "float32"

    # Test with weight and bias (4 inputs)
    inputs = op.decompose(output_tensor, num_inputs=4)
    assert len(inputs) == 4
    # Bias tensor should match normalized_shape
    if normalized_shape:
        assert inputs[3].size == normalized_shape
        assert inputs[3].dtype == "float32"


def test_layer_norm_codegen():
    """Test layer norm code generation."""
    op = LayerNormOperator()

    # Create a dummy output tensor for testing
    output_tensor = Tensor((32, 768), (768, 1), "float32", "cpu", set())

    # Set up normalized_shape for testing
    op._normalized_shape = (768,)

    # Test with minimum inputs
    code = op.codegen("output", ["input", "shape"], output_tensor)
    assert "torch.nn.functional.layer_norm(input, (768,))" in code

    # Test with weight
    code = op.codegen("output", ["input", "shape", "weight"], output_tensor)
    assert "torch.nn.functional.layer_norm(input, (768,), weight=weight)" in code

    # Test with weight and bias
    code = op.codegen("output", ["input", "shape", "weight", "bias"], output_tensor)
    assert "torch.nn.functional.layer_norm(input, (768,), weight=weight, bias=bias)" in code


def test_layer_norm_supports_variable_inputs():
    """Test that layer norm supports variable inputs."""
    op = LayerNormOperator()
    assert op.supports_variable_inputs()
