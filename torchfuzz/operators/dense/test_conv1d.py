"""Test for 1D convolution operator."""

import torch
from torchfuzz.tensor import Tensor
from .conv1d import Conv1dOperator


def test_conv1d_can_produce():
    """Test that Conv1dOperator can produce appropriate tensors."""
    op = Conv1dOperator()
    
    # Should only produce 3D tensors (batch, out_channels, length)
    assert not op.can_produce(Tensor((10,), (1,), "float32", "cpu", set()))
    assert not op.can_produce(Tensor((5, 10), (10, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((2, 16, 128), (2048, 128, 1), "float32", "cpu", set()))
    assert op.can_produce(Tensor((8, 64, 256), (16384, 256, 1), "float32", "cpu", set()))
    assert not op.can_produce(Tensor((3, 2, 5, 10), (100, 50, 10, 1), "float32", "cpu", set()))


def test_conv1d_decompose():
    """Test conv1d decomposition."""
    op = Conv1dOperator()
    
    # Test with minimum inputs (2)
    output_tensor = Tensor((4, 32, 64), (2048, 64, 1), "float32", "cpu", set())
    inputs = op.decompose(output_tensor, num_inputs=2)
    
    assert len(inputs) == 2
    # Input: (batch_size, in_channels, in_length)
    assert inputs[0].size[0] == 4  # batch_size matches
    assert inputs[0].size[1] > 0  # in_channels > 0
    assert inputs[0].size[2] > 0  # in_length > 0
    assert inputs[0].dtype == "float32"
    # Weight: (out_channels, in_channels, kernel_size)
    assert inputs[1].size[0] == 32  # out_channels matches
    assert inputs[1].size[1] == inputs[0].size[1]  # in_channels matches
    assert inputs[1].size[2] > 0  # kernel_size > 0
    assert inputs[1].dtype == "float32"
    
    # Test with bias (3 inputs)
    inputs = op.decompose(output_tensor, num_inputs=3)
    assert len(inputs) == 3
    # Bias: (out_channels,)
    assert inputs[2].size == (32,)  # out_channels
    assert inputs[2].dtype == "float32"


def test_conv1d_codegen():
    """Test conv1d code generation."""
    op = Conv1dOperator()
    
    # Set up parameters for testing
    op._stride = 2
    op._padding = 1
    
    # Test with minimum inputs
    code = op.codegen("output", ["input", "weight"], None)
    expected = "output = torch.nn.functional.conv1d(input, weight, stride=2, padding=1)"
    assert code == expected
    
    # Test with bias
    code = op.codegen("output", ["input", "weight", "bias"], None)
    expected = "output = torch.nn.functional.conv1d(input, weight, bias, stride=2, padding=1)"
    assert code == expected


def test_conv1d_supports_variable_inputs():
    """Test that conv1d supports variable inputs."""
    op = Conv1dOperator()
    assert op.supports_variable_inputs()


def test_conv1d_valid_dimensions():
    """Test that conv1d creates valid input dimensions."""
    op = Conv1dOperator()
    
    # Test with larger output to ensure valid input length calculation
    output_tensor = Tensor((2, 16, 100), (1600, 100, 1), "float16", "cuda", set())
    inputs = op.decompose(output_tensor, num_inputs=2)
    
    # Verify input tensor dimensions are reasonable
    batch_size, in_channels, in_length = inputs[0].size
    out_channels, weight_in_channels, kernel_size = inputs[1].size
    
    assert batch_size == 2
    assert in_channels > 0
    assert in_length >= kernel_size  # Input length should be at least kernel size
    assert out_channels == 16
    assert weight_in_channels == in_channels
    assert kernel_size > 0
    
    # Check stored parameters
    stride = getattr(op, '_stride', 1)
    padding = getattr(op, '_padding', 0)
    assert stride > 0
    assert padding >= 0